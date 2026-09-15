"""Compute R-Value for the current season and rank teams by aggregate value."""
import sys
from pathlib import Path

import pandas as pd

from rvalue import config, history

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    season = config.SEASON
    weights = config.load_weights()
    print(f"R-Value v2 pipeline — season {season}, weights {weights}\n")

    players = history.season_players(season, weights)
    teams = history.season_teams(season, players)
    players_r = history.player_rvalues(season, players)

    pcols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PS", "AVAIL", "WPS",
             "salary", "effective_cost", "cost_index", "salary_imputed", "two_way",
             "ELIGIBLE", "R_VALUE"]
    players_out = players_r[pcols].sort_values("R_VALUE", ascending=False)
    players_out.to_csv(OUT / "player_rvalue.csv", index=False)

    tcols = ["team", "TeamName", "Conference", "PlayoffRank", "WINS", "LOSSES",
             "payroll", "tax", "total_expenditure", "repeater", "over_tax",
             "over_first_apron", "over_second_apron", "TEAM_PERF", "TOP2_WPS",
             "PO_W", "champion", "TEAM_R_VALUE"]
    teams_out = teams[tcols].sort_values("TEAM_R_VALUE", ascending=False)
    teams_out.to_csv(OUT / "team_rvalue.csv", index=False)

    pd.set_option("display.width", 180)
    print("=== TOP 20 PLAYERS BY R-VALUE (min 500 minutes) ===")
    top = players_out[players_out["MIN"] >= 500].head(20).copy()
    top["salary"] = (top["salary"] / 1e6).round(2)
    top[["WPS", "R_VALUE"]] = top[["WPS", "R_VALUE"]].round(1)
    print(top[["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "WPS", "salary", "R_VALUE"]]
          .rename(columns={"salary": "salary_$M"}).to_string(index=False))

    print("\n=== TEAM R-VALUE RANKING ===")
    t = teams_out.copy()
    for c in ("payroll", "tax", "total_expenditure"):
        t[c] = (t[c] / 1e6).round(1)
    t["TEAM_R_VALUE"] = t["TEAM_R_VALUE"].round(1)
    print(t[["team", "WINS", "LOSSES", "payroll", "tax", "total_expenditure",
             "TEAM_R_VALUE"]]
          .rename(columns={"payroll": "payroll_$M", "tax": "tax_$M",
                           "total_expenditure": "total_spend_$M"}).to_string(index=False))

    best = teams_out.iloc[0]
    print(f"\n>>> Most R-Valuable team, {season}: {best['TeamName']} ({best['team']}) — "
          f"Team R-Value {best['TEAM_R_VALUE']:.1f}, record "
          f"{best['WINS']:.0f}-{best['LOSSES']:.0f}, "
          f"total spend ${best['total_expenditure']/1e6:.1f}M")


if __name__ == "__main__":
    sys.exit(main())
