"""R-Value v2: performance pillars, CBA cost model, player & team R-Value.

v2 changes over v1:
- Box pillar blends per-36 rate with total volume (fixes low-minute per-36 bias).
- Pillar z-scores are stored so weights can be calibrated against outcomes.
- Luxury tax uses the correct CBA schedule for each league year.
"""
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config


# ---------------- name matching ----------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in s.lower())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    # merge runs of single-letter initials: "t j mcconnell" -> "tj mcconnell"
    merged: list[str] = []
    for p in parts:
        if len(p) == 1 and merged and len(merged[-1]) <= 2 and all(
                len(x) == 1 for x in merged[-1]):
            merged[-1] += p
        else:
            merged.append(p)
    return " ".join(merged)


# ---------------- luxury tax ----------------

def luxury_tax(payroll: float, season: str, team: str) -> float:
    cba = config.CBA[season]
    over = payroll - cba["tax"]
    if over <= 0:
        return 0.0
    rates = cba["rates_rep"] if team in cba["repeaters"] else cba["rates_std"]
    width, step = cba["width"], cba["step"]
    tax, i = 0.0, 0
    while over > 0:
        rate = rates[i] if i < len(rates) else rates[-1] + step * (i - len(rates) + 1)
        chunk = min(over, width)
        tax += chunk * rate
        over -= chunk
        i += 1
    return tax


# ---------------- performance ----------------

def _z(series: pd.Series, calib_mask: pd.Series) -> pd.Series:
    mu = series[calib_mask].mean()
    sd = series[calib_mask].std(ddof=0)
    return (series - mu) / (sd if sd > 0 else 1.0)


def build_performance(base: pd.DataFrame, adv: pd.DataFrame, est: pd.DataFrame,
                      weights: dict | None = None,
                      season: str | None = None) -> pd.DataFrame:
    """Player table with pillar z-scores, CPI, PS, availability, WPS."""
    weights = weights or config.load_weights()
    df = base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
               "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]].copy()
    adv_cols = adv[["PLAYER_ID", "OFF_RATING", "DEF_RATING", "TS_PCT", "USG_PCT", "PIE"]]
    df = df.merge(adv_cols, on="PLAYER_ID", how="left")
    est_cols = est[["PLAYER_ID", "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING"]]
    df = df.merge(est_cols, on="PLAYER_ID", how="left")
    df = df[df["MIN"] > 0].reset_index(drop=True)

    minw = df["MIN"] / df["MIN"].sum()
    lg_off = float((df["OFF_RATING"] * minw).sum())
    lg_def = float((df["DEF_RATING"] * minw).sum())
    lg_eoff = float((df["E_OFF_RATING"] * minw).sum())
    lg_edef = float((df["E_DEF_RATING"] * minw).sum())
    lg_ts = float((df["TS_PCT"] * minw).sum())

    bw = config.BOX_WEIGHTS
    box_total = (bw["PTS"] * df["PTS"] + bw["REB"] * df["REB"] + bw["AST"] * df["AST"]
                 + bw["STL"] * df["STL"] + bw["BLK"] * df["BLK"] + bw["TOV"] * df["TOV"])
    df["BOX36"] = box_total / df["MIN"] * 36.0
    df["BOX_TOTAL"] = box_total
    df["EFF"] = (df["TS_PCT"] - lg_ts) * df["USG_PCT"]

    rel = df["MIN"] / (df["MIN"] + config.SHRINK_MIN)
    df["OFF_IMP"] = (0.6 * (df["E_OFF_RATING"] - lg_eoff)
                     + 0.4 * (df["OFF_RATING"] - lg_off)) * rel
    df["DEF_IMP"] = (0.6 * (lg_edef - df["E_DEF_RATING"])
                     + 0.4 * (lg_def - df["DEF_RATING"])) * rel

    qual = df["MIN"] >= config.QUALIFIED_MIN
    # v2: box pillar = half rate, half volume
    df["z_box"] = 0.5 * _z(df["BOX36"], qual) + 0.5 * _z(df["BOX_TOTAL"], qual)
    df["z_eff"] = _z(df["EFF"], qual)
    df["z_off"] = _z(df["OFF_IMP"], qual)
    df["z_dfn"] = _z(df["DEF_IMP"], qual)
    df["z_adv"] = _z(df["PIE"], qual)
    # players missing an input (e.g. absent from estimated metrics) get a
    # neutral pillar rather than poisoning downstream aggregation with NaN
    zcols = ["z_box", "z_eff", "z_off", "z_dfn", "z_adv"]
    df[zcols] = df[zcols].fillna(0.0)

    avail = np.sqrt(np.minimum(df["MIN"], config.FULL_CREDIT_MIN) / config.FULL_CREDIT_MIN)
    df["AVAIL"] = avail
    df["ELIGIBLE"] = df["GP"] >= config.gp_threshold(season)
    return apply_weights(df, weights)


def apply_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """(Re)compute CPI, PS, WPS from stored pillar z-scores."""
    df = df.copy()
    df["CPI"] = (weights["box"] * df["z_box"] + weights["eff"] * df["z_eff"]
                 + weights["off"] * df["z_off"] + weights["dfn"] * df["z_dfn"]
                 + weights["adv"] * df["z_adv"])
    df["PS"] = 100.0 * norm.cdf(df["CPI"])
    df["WPS"] = df["PS"] * (config.AVAIL_FLOOR + (1 - config.AVAIL_FLOOR) * df["AVAIL"])
    return df


# ---------------- cost & R-Value ----------------

def build_team_costs(salaries: pd.DataFrame, season: str,
                     payroll_agg: pd.DataFrame | None = None) -> pd.DataFrame:
    sal = salaries[~salaries["two_way"]].copy()
    sal["cost"] = sal[["salary", "cap_allocation"]].max(axis=1)
    teams = (sal.groupby("team", as_index=False)
                .agg(payroll=("cost", "sum"), contracts=("cost", "size")))
    if payroll_agg is not None:
        # outer merge: the server-side aggregate always covers all 30 teams,
        # even when the per-player crawl found no rows for one
        teams = teams.merge(payroll_agg, on="team", how="outer")
        teams["contracts"] = teams["contracts"].fillna(0)
        # prefer the aggregate when larger (it can include dead-money rows
        # the per-player scrape misses)
        teams["payroll"] = teams[["payroll", "payroll_agg"]].max(axis=1)
        teams = teams.drop(columns="payroll_agg")
    cba = config.CBA[season]
    teams["repeater"] = teams["team"].isin(cba["repeaters"])
    teams["tax"] = [luxury_tax(p, season, t) for p, t in zip(teams["payroll"], teams["team"])]
    teams["total_expenditure"] = teams["payroll"] + teams["tax"]
    teams["over_tax"] = teams["payroll"] > cba["tax"]
    teams["over_first_apron"] = teams["payroll"] > cba["apron1"]
    teams["over_second_apron"] = teams["payroll"] > cba["apron2"]
    return teams


def attach_salaries(perf: pd.DataFrame, salaries: pd.DataFrame,
                    season: str | None = None) -> pd.DataFrame:
    df = perf.copy()
    df["norm_name"] = df["PLAYER_NAME"].map(norm_name)

    sal = salaries.copy()
    sal["norm_name"] = sal["player_name"].map(norm_name)
    sal["norm_name"] = sal["norm_name"].replace(config.NAME_ALIASES)
    # total pay across all paying teams for the season (traded players)
    pay = (sal[~sal["two_way"]]
           .groupby("norm_name", as_index=False)["salary"].sum())
    two_way = sal.groupby("norm_name")["two_way"].all().rename("two_way")

    df = df.merge(pay, on="norm_name", how="left").merge(two_way, on="norm_name", how="left")
    df["two_way"] = df["two_way"].astype("boolean").fillna(False).astype(bool)
    df["salary_imputed"] = df["salary"].isna() | (df["salary"] <= 0)
    if season is not None:
        hh = config.hh_season(season)
        known = df["norm_name"].map(
            lambda n: config.KNOWN_SALARIES.get((n, hh)))
        patch = df["salary_imputed"] & known.notna()
        df.loc[patch, "salary"] = known[patch]
        df.loc[patch, "salary_imputed"] = False
    # impute a years-of-service minimum, using age as a YOS proxy (drafted ~22);
    # the minimum scale is indexed to that season's cap
    yos = (df["AGE"] - 22).clip(0, 10).fillna(2).astype(int)
    era = (config.CBA[season]["cap"] / config.CBA[config.SEASON]["cap"]
           if season is not None else 1.0)
    impute = era * config.MIN_SALARY * yos.map(lambda y: config.MIN_SALARY_FACTORS[y])
    df.loc[df["salary_imputed"], "salary"] = impute[df["salary_imputed"]]
    return df


def compute_rvalue(perf_sal: pd.DataFrame, team_costs: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = perf_sal.merge(
        team_costs[["team", "payroll", "tax", "total_expenditure"]],
        left_on="TEAM_ABBREVIATION", right_on="team", how="left")

    league_avg_te = team_costs["total_expenditure"].mean()
    grossup = (df["total_expenditure"] / df["payroll"]).fillna(1.0)
    df["effective_cost"] = df["salary"] * grossup
    lo, hi = config.COST_SHARE_CLIP
    df["cost_index"] = (df["effective_cost"] / league_avg_te).clip(lo, hi)

    mean_ci = df["cost_index"].mean()
    df["R_raw"] = df["WPS"] * (mean_ci / df["cost_index"]) ** config.COST_EXPONENT
    df["R_VALUE"] = 100.0 * df["R_raw"] / df["R_raw"].mean()

    g = df.groupby("TEAM_ABBREVIATION").apply(
        lambda t: (t["WPS"] * t["MIN"]).sum(), include_groups=False
    ).rename("perf_minutes").reset_index()
    teams = team_costs.merge(g, left_on="team", right_on="TEAM_ABBREVIATION", how="left")
    teams["TeamR_raw"] = teams["perf_minutes"] / teams["total_expenditure"]
    teams["TEAM_R_VALUE"] = 100.0 * teams["TeamR_raw"] / teams["TeamR_raw"].mean()

    # payroll utilization: share of payroll paid to real rotation players
    # (>=500 minutes); the rest bought bench decoration or dead money
    active = (df[df["MIN"] >= 500].groupby("TEAM_ABBREVIATION")["salary"].sum()
              .rename("active_salary").reset_index())
    teams = teams.merge(active, on="TEAM_ABBREVIATION", how="left")
    teams["active_dollar_share"] = (teams["active_salary"] / teams["payroll"]).clip(0, 1)
    teams = teams.drop(columns="active_salary")
    return df, teams


def team_performance(players: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted team performance core (cost-free) plus rotation shape.

    TEAM_PERF: quality per minute over everyone who played.
    TOP2_WPS:  star ceiling (best two players).
    ROT8_WPS:  the de facto rotation - the eight most-played players,
               minutes-weighted (playoff rotations compress to ~8).
    BENCH_WPS: players 9-13 by minutes (depth behind the rotation).
    """
    def agg(t):
        perf = (t["WPS"] * t["MIN"]).sum() / t["MIN"].sum()
        top2 = t.nlargest(2, "WPS")["WPS"].mean()
        rot = t.nlargest(8, "MIN")
        rot8 = (rot["WPS"] * rot["MIN"]).sum() / rot["MIN"].sum()
        bench = t.nlargest(13, "MIN").nsmallest(5, "MIN")
        bench_w = (bench["WPS"] * bench["MIN"]).sum() / max(bench["MIN"].sum(), 1.0)
        return pd.Series({"TEAM_PERF": perf, "TOP2_WPS": top2,
                          "ROT8_WPS": rot8, "BENCH_WPS": bench_w})
    return (players.groupby("TEAM_ABBREVIATION").apply(agg, include_groups=False)
            .reset_index())
