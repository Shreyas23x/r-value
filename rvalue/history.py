"""Assemble per-season datasets: performance, costs, standings, champion."""
import pandas as pd

from . import config, fetch, model


def season_players(season: str, weights: dict | None = None) -> pd.DataFrame:
    base = fetch.fetch_player_base(season)
    adv = fetch.fetch_player_advanced(season)
    est = fetch.fetch_player_estimated(season)
    return model.build_performance(base, adv, est, weights, season)


def _history_supplement() -> pd.DataFrame:
    """Backfill salaries purged from team pages: identify >=500-minute players
    in any history season whose salary row is missing, then pull their
    individual salary-history pages (needs an hh_player_id from the crawl)."""
    multi = fetch.fetch_contracts_multi(config.HISTORY_SEASONS + [config.PREDICT_SEASON])
    multi_nn = multi.copy()
    multi_nn["nn"] = (multi_nn["player_name"].map(model.norm_name)
                      .replace(config.NAME_ALIASES))

    needed = []
    for season in config.HISTORY_SEASONS:
        hh = config.hh_season(season)
        base = fetch.fetch_player_base(season)
        stats = base[base["MIN"] >= config.QUALIFIED_MIN].copy()
        stats["nn"] = stats["PLAYER_NAME"].map(model.norm_name)
        have = set(multi_nn.loc[multi_nn["hh_season"] == hh, "nn"])
        # dedicated per-season crawl covers players whose contract was never
        # superseded (including retirees and departures)
        ded = fetch.fetch_salaries(season)
        have |= set(ded["player_name"].map(model.norm_name)
                    .replace(config.NAME_ALIASES))
        needed.extend(stats.loc[~stats["nn"].isin(have), "nn"])

    ids = (multi_nn[multi_nn["nn"].isin(set(needed))]
           [["hh_player_id", "player_name"]].drop_duplicates("hh_player_id"))
    hh_seasons = {config.hh_season(s) for s in config.HISTORY_SEASONS}
    return fetch.fetch_player_histories(ids, hh_seasons)


def season_salaries(season: str) -> pd.DataFrame:
    """Best-available per-player salary rows for a season.

    Merges: dedicated per-season crawl (current season only), the
    multi-season pass over current contract records, and the player-page
    history backfill for re-signed players.
    """
    hh = config.hh_season(season)
    frames = []
    df = fetch.fetch_salaries(season).copy()
    if "hh_season" not in df.columns:
        df["hh_season"] = hh
    frames.append(df)
    multi = fetch.fetch_contracts_multi(config.HISTORY_SEASONS + [config.PREDICT_SEASON])
    frames.append(multi[multi["hh_season"] == hh])
    supp = _history_supplement()
    if len(supp):
        frames.append(supp[supp["hh_season"] == hh])
    sal = pd.concat(frames, ignore_index=True)
    return (sal.groupby(["player_name", "team"], as_index=False)
               .agg(salary=("salary", "max"), cap_allocation=("cap_allocation", "max"),
                    two_way=("two_way", "any"), terminated=("terminated", "any")))


def season_teams(season: str, players: pd.DataFrame) -> pd.DataFrame:
    """Team table: costs, standings, playoffs, performance core, R-Value."""
    salaries = season_salaries(season)
    payroll_agg = fetch.fetch_team_payrolls(season)
    costs = model.build_team_costs(salaries, season, payroll_agg)

    perf_sal = model.attach_salaries(players, salaries, season)
    _, teams = model.compute_rvalue(perf_sal, costs)
    teams = teams.merge(model.team_performance(players), on="TEAM_ABBREVIATION", how="left")

    st = fetch.fetch_standings(season)
    abbr = (players[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
            .rename(columns={"TEAM_ID": "TeamID"}))
    st = st.merge(abbr, on="TeamID", how="left")
    teams = teams.merge(
        st[["TEAM_ABBREVIATION", "TeamName", "Conference", "PlayoffRank", "WINS", "LOSSES"]],
        on="TEAM_ABBREVIATION", how="left")

    po = fetch.fetch_playoff_wins(season).rename(columns={"TEAM_ID": "TeamID"})
    po = po.merge(abbr, on="TeamID", how="left")
    teams = teams.merge(po[["TEAM_ABBREVIATION", "PO_W", "PO_L"]],
                        on="TEAM_ABBREVIATION", how="left")
    teams[["PO_W", "PO_L"]] = teams[["PO_W", "PO_L"]].fillna(0)
    teams["champion"] = teams["PO_W"] == teams["PO_W"].max()

    net = fetch.fetch_team_net_rating(season).rename(columns={"TEAM_ID": "TeamID"})
    net = net.merge(abbr, on="TeamID", how="left")
    teams = teams.merge(net[["TEAM_ABBREVIATION", "NET_RATING"]],
                        on="TEAM_ABBREVIATION", how="left")
    return teams


def player_rvalues(season: str, players: pd.DataFrame) -> pd.DataFrame:
    """Player-level R-Values for a season (salary coverage permitting)."""
    salaries = season_salaries(season)
    payroll_agg = fetch.fetch_team_payrolls(season)
    costs = model.build_team_costs(salaries, season, payroll_agg)
    perf_sal = model.attach_salaries(players, salaries, season)
    out, _ = model.compute_rvalue(perf_sal, costs)
    return out
