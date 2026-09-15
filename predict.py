"""Predict 2026-27 from calibrated R-Value: seeds, champion, All-NBA, value."""
import sys

import pandas as pd

from rvalue import config, model, project

pd.set_option("display.width", 200)
sys.stdout.reconfigure(encoding="utf-8")  # player names vs Windows cp1252 console


def main() -> None:
    weights = config.load_weights()
    lam = config.load_star_lambda()
    rot_lam = config.load_rot_lambda()
    print(f"Projecting {config.PREDICT_SEASON} with calibrated weights {weights}, "
          f"star lambda {lam}, rotation lambda {rot_lam}\n")

    players = project.project_players(weights)
    known = players["has_history"].sum()
    print(f"{len(players)} players under contract, {known} with NBA history, "
          f"{len(players) - known} rookies/unknown at replacement level")

    teams = project.project_teams(players, lam, rot_lam)
    players_r = project.player_rvalues_proj(players, teams)

    print("\n=== PREDICTED 2026-27 STANDINGS (by projected performance core) ===")
    for conf in ("East", "West"):
        t = (teams[teams["conference"] == conf]
             .sort_values("pred_seed")
             [["pred_seed", "team", "TEAM_PERF", "TITLE_SCORE", "TEAM_R_VALUE",
               "payroll", "tax", "n_players"]].copy())
        t[["payroll", "tax"]] = (t[["payroll", "tax"]] / 1e6).round(1)
        t[["TEAM_PERF", "TITLE_SCORE", "TEAM_R_VALUE"]] = \
            t[["TEAM_PERF", "TITLE_SCORE", "TEAM_R_VALUE"]].round(2)
        print(f"\n{conf}:")
        print(t.rename(columns={"payroll": "payroll_$M", "tax": "tax_$M"})
              .to_string(index=False))

    champ = teams.sort_values("TITLE_SCORE", ascending=False).head(5)
    print("\n=== PREDICTED TITLE CONTENDERS (title score = perf + star + rotation premiums) ===")
    print(champ[["team", "conference", "pred_seed", "TITLE_SCORE", "TEAM_PERF",
                 "TOP2_WPS", "ROT8_WPS"]].round(2).to_string(index=False))

    print("\n=== PREDICTED 2026-27 ALL-NBA (top 15 projected WPS, durability-gated) ===")
    pool = players_r[players_r["MIN_raw"] >= 1600].nlargest(15, "WPS_proj")
    out = pool[["PLAYER_NAME", "team", "AGE_proj", "PS_proj", "WPS_proj",
                "salary", "R_VALUE_proj"]].copy()
    out["salary"] = (out["salary"] / 1e6).round(1)
    out[["PS_proj", "WPS_proj", "R_VALUE_proj"]] = \
        out[["PS_proj", "WPS_proj", "R_VALUE_proj"]].round(1)
    out.insert(0, "all_nba_team", ["1st"] * 5 + ["2nd"] * 5 + ["3rd"] * 5)
    print(out.rename(columns={"salary": "salary_$M"}).to_string(index=False))

    best_value = teams.sort_values("TEAM_R_VALUE", ascending=False)
    print("\n=== PREDICTED MOST R-VALUABLE TEAMS 2026-27 ===")
    bv = best_value.head(8)[["team", "conference", "pred_seed", "TEAM_R_VALUE",
                             "payroll", "total_expenditure"]].copy()
    bv[["payroll", "total_expenditure"]] = (bv[["payroll", "total_expenditure"]] / 1e6).round(1)
    bv["TEAM_R_VALUE"] = bv["TEAM_R_VALUE"].round(1)
    print(bv.to_string(index=False))

    print("\n=== PREDICTED TOP 15 PLAYER R-VALUES 2026-27 (min 1200 proj minutes) ===")
    pv = players_r[players_r["MIN_raw"] >= 1200].nlargest(15, "R_VALUE_proj")
    pv_out = pv[["PLAYER_NAME", "team", "WPS_proj", "salary", "R_VALUE_proj"]].copy()
    pv_out["salary"] = (pv_out["salary"] / 1e6).round(1)
    pv_out[["WPS_proj", "R_VALUE_proj"]] = pv_out[["WPS_proj", "R_VALUE_proj"]].round(1)
    print(pv_out.rename(columns={"salary": "salary_$M"}).to_string(index=False))

    teams.sort_values(["conference", "pred_seed"]).to_csv(
        "output/predicted_teams_2026-27.csv", index=False)
    players_r.sort_values("R_VALUE_proj", ascending=False).to_csv(
        "output/predicted_players_2026-27.csv", index=False)

    top = teams.sort_values("TITLE_SCORE", ascending=False).iloc[0]
    val = best_value.iloc[0]
    print(f"\n>>> Predicted 2026-27 champion: {top['team']} "
          f"(title score {top['TITLE_SCORE']:.2f}, projected {top['conference']} #{top['pred_seed']})")
    print(f">>> Predicted most R-Valuable team 2026-27: {val['team']} "
          f"(Team R-Value {val['TEAM_R_VALUE']:.1f})")


if __name__ == "__main__":
    main()
