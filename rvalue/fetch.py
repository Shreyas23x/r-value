"""Data acquisition: NBA stats via nba_api, salaries via hoopshype embedded JSON.

Everything is cached as CSV under data/ so repeat runs are offline.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def _cached(name: str, builder) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = builder()
    df.to_csv(path, index=False)
    return df


# ---------------- NBA stats ----------------

def fetch_player_base(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import leaguedashplayerstats
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="Totals", timeout=90
        ).get_data_frames()[0]
    return _cached(f"player_base_{season}", build)


def fetch_player_advanced(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import leaguedashplayerstats
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Advanced",
            per_mode_detailed="Totals", timeout=90
        ).get_data_frames()[0]
    return _cached(f"player_advanced_{season}", build)


def fetch_player_estimated(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import playerestimatedmetrics
        return playerestimatedmetrics.PlayerEstimatedMetrics(
            season=season, timeout=90
        ).get_data_frames()[0]
    return _cached(f"player_estimated_{season}", build)


def fetch_standings(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import leaguestandingsv3
        df = leaguestandingsv3.LeagueStandingsV3(season=season, timeout=90
                                                 ).get_data_frames()[0]
        return df[["TeamID", "TeamCity", "TeamName", "Conference",
                   "PlayoffRank", "WINS", "LOSSES"]]
    return _cached(f"standings_{season}", build)


def fetch_playoff_wins(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import leaguedashteamstats
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star="Playoffs", timeout=90
        ).get_data_frames()[0]
        return df[["TEAM_ID", "TEAM_NAME", "W", "L"]].rename(
            columns={"W": "PO_W", "L": "PO_L"})
    return _cached(f"playoffs_{season}", build)


def fetch_team_net_rating(season: str) -> pd.DataFrame:
    def build():
        from nba_api.stats.endpoints import leaguedashteamstats
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced", timeout=90
        ).get_data_frames()[0]
        return df[["TEAM_ID", "TEAM_NAME", "NET_RATING"]]
    return _cached(f"team_netrtg_{season}", build)


# ---------------- Salaries (hoopshype) ----------------

def _next_data(session: requests.Session, url: str, retries: int = 3) -> dict:
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=45)
            r.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
            time.sleep(3.0 * (attempt + 1))
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  r.text, re.S)
    if not m:
        raise RuntimeError(f"No __NEXT_DATA__ on {url}")
    return json.loads(m.group(1))


def _find_contracts(obj):
    if isinstance(obj, dict):
        if obj.get("__typename") == "SearchContracts" and "contracts" in obj:
            return obj["contracts"]
        for v in obj.values():
            got = _find_contracts(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_contracts(v)
            if got is not None:
                return got
    return None


def _contract_rows(contracts: list, seasons: set[int]) -> list[dict]:
    rows = []
    for c in contracts or []:
        for s in c.get("seasons") or []:
            tid = s.get("teamID")
            if s.get("season") in seasons and tid in config.HH_TEAM_ABBR:
                rows.append({
                    "player_name": c.get("playerName", ""),
                    "hh_player_id": c.get("playerID", ""),
                    "hh_season": s.get("season"),
                    "team": config.HH_TEAM_ABBR[tid],
                    "salary": s.get("salary") or 0,
                    "cap_allocation": s.get("capAllocation") or 0,
                    "two_way": bool(s.get("twoWayContract")),
                    "terminated": bool(s.get("terminated")),
                })
    return rows


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["player_name", "team", "hh_season"], as_index=False)
              .agg(salary=("salary", "max"), cap_allocation=("cap_allocation", "max"),
                   two_way=("two_way", "any"), terminated=("terminated", "any"),
                   hh_player_id=("hh_player_id", "first")))


def fetch_salaries(season: str) -> pd.DataFrame:
    """Per-player salary rows for one season, from that season's team pages."""
    hh = config.hh_season(season)

    def build():
        session = requests.Session()
        session.headers.update({"User-Agent": UA})
        all_rows = []
        for slug, tid in config.HH_TEAM_SLUGS.items():
            url = f"https://hoopshype.com/salaries/teams/{slug}/{tid}/?season={hh}"
            payload = _next_data(session, url)
            rows = _contract_rows(
                _find_contracts(payload["props"]["pageProps"]["dehydratedState"]),
                {hh})
            all_rows.extend(rows)
            time.sleep(0.6)
        return _dedupe(pd.DataFrame(all_rows))
    return _cached(f"salaries_{season}", build)


def fetch_contracts_multi(seasons: list[str]) -> pd.DataFrame:
    """One pass over current team pages, keeping rows for several seasons.

    Contract records carry full per-season history for currently signed
    players, so a single crawl yields salaries across recent seasons
    (coverage degrades for players no longer under contract).
    """
    hh_set = {config.hh_season(s) for s in seasons}
    tag = "_".join(sorted(str(s) for s in hh_set))

    def build():
        session = requests.Session()
        session.headers.update({"User-Agent": UA})
        all_rows = []
        for slug, tid in config.HH_TEAM_SLUGS.items():
            url = f"https://hoopshype.com/salaries/teams/{slug}/{tid}/"
            payload = _next_data(session, url)
            rows = _contract_rows(
                _find_contracts(payload["props"]["pageProps"]["dehydratedState"]),
                hh_set)
            all_rows.extend(rows)
            time.sleep(0.6)
        return _dedupe(pd.DataFrame(all_rows))
    return _cached(f"contracts_multi_{tag}", build)


def _player_slug(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
    return "-".join(s.split())


def fetch_player_histories(targets: pd.DataFrame, seasons: set[int]) -> pd.DataFrame:
    """Full salary history from individual player pages.

    targets: DataFrame with hh_player_id and player_name (hoopshype display
    names). Used to backfill past-season salaries that were purged from the
    team-page contract records when players re-signed.
    """
    def build():
        session = requests.Session()
        session.headers.update({"User-Agent": UA})
        all_rows, misses = [], 0
        targets_u = targets.drop_duplicates("hh_player_id")
        print(f"  backfilling salary history for {len(targets_u)} players...")
        for r in targets_u.itertuples():
            slug = _player_slug(r.player_name)
            pid = str(int(float(r.hh_player_id)))
            url = f"https://hoopshype.com/salaries/players/{slug}/{pid}/"
            try:
                payload = _next_data(session, url)
                rows = _contract_rows(
                    _find_contracts(payload["props"]["pageProps"]["dehydratedState"]),
                    seasons)
                for row in rows:
                    row["player_name"] = r.player_name
                all_rows.extend(rows)
            except Exception:
                misses += 1
            time.sleep(0.6)
        if misses:
            print(f"  ({misses} player pages not found; imputation stays for those)")
        if not all_rows:
            return pd.DataFrame(columns=["player_name", "team", "hh_season", "salary",
                                         "cap_allocation", "two_way", "terminated",
                                         "hh_player_id"])
        return _dedupe(pd.DataFrame(all_rows))
    return _cached("player_histories", build)


def fetch_team_payrolls(season: str) -> pd.DataFrame:
    """Server-side per-team payroll aggregates for the season (authoritative)."""
    hh = config.hh_season(season)

    def build():
        session = requests.Session()
        session.headers.update({"User-Agent": UA})
        payload = _next_data(
            session, f"https://hoopshype.com/salaries/teams/?season={hh}")

        rows = []
        def walk(obj):
            if isinstance(obj, dict):
                key = obj.get("key", "")
                if isinstance(key, str) and key.startswith(f"{hh}|"):
                    tid = key.split("|", 1)[1]
                    if tid in config.HH_TEAM_ABBR:
                        sal = cap = 0
                        for agg in obj.get("metricAggregations") or []:
                            if agg.get("name") == "teams_salary_sum":
                                sal = agg.get("value") or 0
                            elif agg.get("name") == "teams_capAllocation_sum":
                                cap = agg.get("value") or 0
                        rows.append({"team": config.HH_TEAM_ABBR[tid],
                                     "payroll_agg": sal + cap})
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(payload["props"]["pageProps"]["dehydratedState"])
        return pd.DataFrame(rows).groupby("team", as_index=False)["payroll_agg"].max()
    return _cached(f"team_payrolls_{season}", build)
