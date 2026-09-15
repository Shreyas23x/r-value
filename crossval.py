"""Leave-one-season-out cross-validation of R-Value's fitted parameters.

backtest.py reports accuracy on the same seasons it calibrated against, which
overstates it. This script refits the pillar weights and the title premiums on
every season but one, scores the held-out season, and rotates. Those are the
numbers to quote publicly.

Rival metrics (PIE x minutes, plus-minus, estimated net rating) have no fitted
parameters, so their figures are already out-of-sample and carry over as-is.
"""
import json
import sys

import pandas as pd

from rvalue import calibrate, config, history

pd.set_option("display.width", 200)
sys.stdout.reconfigure(encoding="utf-8")  # player names vs Windows cp1252 console


def main() -> None:
    print("Building season datasets (cached)...")
    season_players, season_teams = {}, {}
    for s in config.HISTORY_SEASONS:
        p = history.season_players(s, config.DEFAULT_WEIGHTS)
        season_players[s], season_teams[s] = p, history.season_teams(s, p)

    n = len(config.HISTORY_SEASONS)
    print(f"\nLeave-one-season-out over {n} seasons "
          f"(refitting weights + title premiums on {n - 1} seasons per fold)...")
    cv = calibrate.cross_validate(season_players, season_teams)
    cv.to_csv("output/cross_validation.csv", index=False)

    print("\n=== OUT-OF-SAMPLE PERFORMANCE BY HELD-OUT SEASON ===")
    print(cv[["held_out", "recall@15", "spearman_wins", "champ_title_rank",
              "star_lambda", "rot_lambda"]].round(3).to_string(index=False))

    print("\n=== WEIGHTS CHOSEN PER FOLD (stability check) ===")
    wcols = [c for c in cv.columns if c.startswith("w_")]
    print(cv[["held_out"] + wcols].round(2).to_string(index=False))
    spread = {c[2:]: round(cv[c].max() - cv[c].min(), 2) for c in wcols}
    print(f"  spread across folds (max-min): {spread}")

    # rival metrics need weights only for the R-Value row, which we ignore here
    full = json.loads(config.WEIGHTS_FILE.read_text())
    player_cmp, _ = calibrate.metric_comparison(
        season_players, season_teams, full["weights"])
    rivals = player_cmp["mean"].drop(index="R-Value (WPS core)", errors="ignore")

    print("\n=== HEADLINE: IN-SAMPLE vs OUT-OF-SAMPLE ===")
    rows = [
        {"metric": "R-Value, in-sample (backtest.py)",
         "recall@15": player_cmp.loc["R-Value (WPS core)", "mean"],
         "spearman_wins": float("nan"),
         "champ_mean_rank": full["champion_mean_title_rank"]},
        {"metric": "R-Value, out-of-sample (LOSO)",
         "recall@15": cv["recall@15"].mean(),
         "spearman_wins": cv["spearman_wins"].mean(),
         "champ_mean_rank": cv["champ_title_rank"].mean()},
    ]
    for name, val in rivals.items():
        rows.append({"metric": f"{name} (no fitted params)", "recall@15": val,
                     "spearman_wins": float("nan"), "champ_mean_rank": float("nan")})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print(f"\n  out-of-sample recall@15  : {cv['recall@15'].mean():.3f}")
    print(f"  out-of-sample wins rho   : {cv['spearman_wins'].mean():.3f}")
    print(f"  out-of-sample champ rank : {cv['champ_title_rank'].mean():.2f} "
          f"(of 30 teams)")
    print("  -> quote these, not backtest.py's, when the numbers are public")


if __name__ == "__main__":
    main()
