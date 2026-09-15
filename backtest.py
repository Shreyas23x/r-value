"""Backtest & calibrate R-Value on 2023-24 .. 2025-26, then report:
- calibrated weights (maximizing All-NBA recall + wins correlation)
- R-Value vs rival metrics comparison
- champions / top seeds / All-NBA players through the R-Value lens
"""
import sys

import numpy as np
import pandas as pd

from rvalue import calibrate, config, history, model

pd.set_option("display.width", 200)
sys.stdout.reconfigure(encoding="utf-8")  # player names vs Windows cp1252 console


def main() -> None:
    print("Building season datasets (first run fetches & caches)...")
    season_players, season_teams = {}, {}
    for s in config.HISTORY_SEASONS:
        p = history.season_players(s, config.DEFAULT_WEIGHTS)
        t = history.season_teams(s, p)
        season_players[s], season_teams[s] = p, t
        print(f"  {s}: {len(p)} players, champion = "
              f"{t.loc[t['champion'], 'TeamName'].iloc[0]}")

    print("\nCalibrating weights (two-stage simplex grid search)...")
    result = calibrate.calibrate(season_players, season_teams)
    w = result["weights"]
    print(f"  calibrated weights: box={w['box']:.2f} eff={w['eff']:.2f} "
          f"off={w['off']:.2f} def={w['dfn']:.2f} adv={w['adv']:.2f}")
    print(f"  star premium lambda={result['star_lambda']:.1f}, "
          f"rotation premium lambda={result['rot_lambda']:.1f}, "
          f"objective={result['objective']:.4f}, "
          f"champion mean title-score rank={result['champion_mean_title_rank']:.2f}")

    # rebuild everything with calibrated weights
    for s in config.HISTORY_SEASONS:
        season_players[s] = model.apply_weights(season_players[s], w)
        season_teams[s] = history.season_teams(s, season_players[s])

    print("\n=== METRIC COMPARISON: predicting All-NBA (recall@15, players) ===")
    pc, tc = calibrate.metric_comparison(season_players, season_teams, w)
    print(pc.round(3).to_string())
    print("\n=== METRIC COMPARISON: predicting wins (spearman, teams) ===")
    print(tc.round(3).to_string())

    print("\n=== CHAMPIONS & SEEDS THROUGH THE R-VALUE LENS ===")
    rows = []
    for s in config.HISTORY_SEASONS:
        t = season_teams[s].copy()
        t["perf_rank"] = t["TEAM_PERF"].rank(ascending=False)
        t["rvalue_rank"] = t["TEAM_R_VALUE"].rank(ascending=False)
        t["title_score"] = (calibrate._zs(t["TEAM_PERF"])
                            + result["star_lambda"] * calibrate._zs(t["TOP2_WPS"])
                            + result["rot_lambda"] * calibrate._zs(t["ROT8_WPS"]))
        t["title_rank"] = t["title_score"].rank(ascending=False)
        ch = t[t["champion"]].iloc[0]
        one_seeds = t[t["PlayoffRank"] == 1]
        rows.append({
            "season": s, "champion": ch["TeamName"],
            "champ_title_rank": int(ch["title_rank"]),
            "champ_perf_rank": int(ch["perf_rank"]),
            "champ_RValue": round(ch["TEAM_R_VALUE"], 1),
            "champ_RValue_rank": int(ch["rvalue_rank"]),
            "1-seeds": ", ".join(f"{r.TEAM_ABBREVIATION}(perf#{int(r.perf_rank)},R={r.TEAM_R_VALUE:.0f})"
                                  for r in one_seeds.itertuples()),
        })
        t.to_csv(f"output/team_rvalue_{s}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== ALL-NBA SELECTIONS: R-VALUE VIEW (per season) ===")
    for s in config.HISTORY_SEASONS:
        p = history.player_rvalues(s, season_players[s])
        p["nn"] = p["PLAYER_NAME"].map(model.norm_name)
        truth = {model.norm_name(n) for n in config.ALL_NBA[s]}
        sel = p[p["nn"].isin(truth)].sort_values("WPS", ascending=False)
        elig = p[p["ELIGIBLE"]]
        n_top15 = (elig.nlargest(15, "WPS")["nn"].isin(truth)).sum()
        print(f"\n{s}: {n_top15}/15 actual All-NBA appear in R-Value's top 15 (eligible)")
        cols = sel[["PLAYER_NAME", "TEAM_ABBREVIATION", "PS", "WPS", "salary", "R_VALUE"]].copy()
        cols["salary"] = (cols["salary"] / 1e6).round(1)
        cols[["PS", "WPS", "R_VALUE"]] = cols[["PS", "WPS", "R_VALUE"]].round(1)
        print(cols.rename(columns={"salary": "salary_$M"}).to_string(index=False))
        p.to_csv(f"output/player_rvalue_{s}.csv", index=False)


if __name__ == "__main__":
    main()
