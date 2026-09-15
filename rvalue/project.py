"""Project R-Value into 2026-27: predicted seeds, champion, All-NBA, value.

Method:
- Rosters & salaries: current contract records for 2026-27 (players signed as
  of today; unsigned FAs and undrafted rookies are absent -> replacement level).
- Player projection: blend of last three seasons' CPI (0.55/0.30/0.15,
  renormalized over available seasons) plus an age-curve adjustment.
- Minutes projection: blend of last two seasons, floored, then rescaled so
  each roster totals 19,680 player-minutes (82 games x 240).
- Team performance = minutes-weighted projected WPS; title score adds the
  calibrated star and rotation premiums; Team R-Value divides by projected
  total spend.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config, fetch, history, model

SEASON_BLEND = {"2025-26": 0.55, "2024-25": 0.30, "2023-24": 0.15}
ROOKIE_CPI = -0.8            # replacement-ish level for unknown/rookie players
ROOKIE_MIN = 900
TEAM_MINUTES = 82 * 240


def _age_adjust(age: float) -> float:
    if age <= 23: return 0.12
    if age <= 25: return 0.06
    if age <= 29: return 0.0
    if age <= 32: return -0.08
    return -0.18


def project_players(weights: dict) -> pd.DataFrame:
    """One row per player signed for 2026-27, with projected CPI/minutes."""
    hist = {}
    for s in config.HISTORY_SEASONS:
        p = history.season_players(s, weights)
        p["nn"] = p["PLAYER_NAME"].map(model.norm_name)
        hist[s] = p.set_index("nn")

    multi = fetch.fetch_contracts_multi(config.HISTORY_SEASONS + [config.PREDICT_SEASON])
    hh = config.hh_season(config.PREDICT_SEASON)
    roster = multi[(multi["hh_season"] == hh) & (~multi["two_way"])
                   & (multi["salary"] > 0) & (~multi["terminated"])].copy()
    roster["nn"] = (roster["player_name"].map(model.norm_name)
                    .replace(config.NAME_ALIASES))
    # a player signed via sign-and-trade may show rows on two teams: keep the
    # highest-salary row as the playing team, but sum salary for his cost
    pay = roster.groupby("nn", as_index=False)["salary"].sum()
    roster = (roster.sort_values("salary", ascending=False)
              .drop_duplicates("nn")[["nn", "player_name", "team"]]
              .merge(pay, on="nn"))

    rows = []
    for r in roster.itertuples():
        cpis, mins, wsum = [], [], 0.0
        age = np.nan
        for s, w in SEASON_BLEND.items():
            h = hist[s]
            if r.nn in h.index:
                rec = h.loc[r.nn]
                if isinstance(rec, pd.DataFrame):
                    rec = rec.iloc[-1]
                cpis.append(w * rec["CPI"])
                wsum += w
                if s in ("2025-26", "2024-25"):
                    mins.append(rec["MIN"])
                if s == "2025-26":
                    age = rec["AGE"] + 1
                elif np.isnan(age):
                    age = rec["AGE"] + (2 if s == "2024-25" else 3)
        if wsum > 0:
            cpi = sum(cpis) / wsum + _age_adjust(age if not np.isnan(age) else 27)
            min_proj = (0.65 * mins[0] + 0.35 * mins[1]) if len(mins) == 2 else \
                       (mins[0] if mins else 600.0)
            min_proj = max(min_proj, 300.0)
            known = True
        else:
            cpi, min_proj, known = ROOKIE_CPI, ROOKIE_MIN, False
        rows.append({"nn": r.nn, "PLAYER_NAME": r.player_name, "team": r.team,
                     "salary": r.salary, "CPI_proj": cpi, "MIN_proj": min_proj,
                     "AGE_proj": age, "has_history": known})
    df = pd.DataFrame(rows)

    # individual availability uses the raw projection (a star's durability is
    # not diluted by his team's roster being incomplete mid-free-agency);
    # team aggregation uses minutes normalized to a full season
    df["MIN_raw"] = df["MIN_proj"].clip(upper=3100)
    df["MIN_proj"] = (df["MIN_raw"]
                      * df["team"].map(TEAM_MINUTES / df.groupby("team")["MIN_raw"].sum()))

    df["PS_proj"] = 100.0 * norm.cdf(df["CPI_proj"])
    avail = np.sqrt(np.minimum(df["MIN_raw"], config.FULL_CREDIT_MIN)
                    / config.FULL_CREDIT_MIN)
    df["WPS_proj"] = df["PS_proj"] * (config.AVAIL_FLOOR
                                      + (1 - config.AVAIL_FLOOR) * avail)
    return df


def project_teams(players: pd.DataFrame, star_lambda: float,
                  rot_lambda: float) -> pd.DataFrame:
    season = config.PREDICT_SEASON
    payroll = fetch.fetch_team_payrolls(season).rename(columns={"payroll_agg": "payroll"})
    payroll["tax"] = [model.luxury_tax(p, season, t)
                      for p, t in zip(payroll["payroll"], payroll["team"])]
    payroll["total_expenditure"] = payroll["payroll"] + payroll["tax"]

    def agg(t):
        perf = (t["WPS_proj"] * t["MIN_proj"]).sum() / t["MIN_proj"].sum()
        rot = t.nlargest(8, "MIN_proj")
        return pd.Series({
            "TEAM_PERF": perf,
            "TOP2_WPS": t.nlargest(2, "WPS_proj")["WPS_proj"].mean(),
            "ROT8_WPS": (rot["WPS_proj"] * rot["MIN_proj"]).sum() / rot["MIN_proj"].sum(),
            "perf_minutes": (t["WPS_proj"] * t["MIN_proj"]).sum(),
            "n_players": len(t),
        })
    teams = players.groupby("team").apply(agg, include_groups=False).reset_index()
    teams = teams.merge(payroll, on="team", how="left")

    z = lambda x: (x - x.mean()) / x.std(ddof=0)
    teams["TITLE_SCORE"] = (z(teams["TEAM_PERF"]) + star_lambda * z(teams["TOP2_WPS"])
                            + rot_lambda * z(teams["ROT8_WPS"]))
    teams["TeamR_raw"] = teams["perf_minutes"] / teams["total_expenditure"]
    teams["TEAM_R_VALUE"] = 100.0 * teams["TeamR_raw"] / teams["TeamR_raw"].mean()
    teams["conference"] = teams["team"].map(config.CONFERENCE)
    teams["pred_seed"] = teams.groupby("conference")["TEAM_PERF"] \
                              .rank(ascending=False).astype(int)
    return teams


def player_rvalues_proj(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    df = players.merge(teams[["team", "payroll", "total_expenditure"]], on="team")
    league_avg_te = teams["total_expenditure"].mean()
    grossup = df["total_expenditure"] / df["payroll"]
    lo, hi = config.COST_SHARE_CLIP
    df["cost_index"] = (df["salary"] * grossup / league_avg_te).clip(lo, hi)
    mean_ci = df["cost_index"].mean()
    df["R_raw"] = df["WPS_proj"] * (mean_ci / df["cost_index"]) ** config.COST_EXPONENT
    df["R_VALUE_proj"] = 100.0 * df["R_raw"] / df["R_raw"].mean()
    return df
