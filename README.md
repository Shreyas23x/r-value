# R-Value v2: Return-on-Roster-Investment for NBA Players & Teams

**R-Value** measures how much on-court value a player delivers *per unit of his team's
total expenditure under the 2023 CBA*. League average is calibrated to **100**.
Aggregated to teams, it identifies which franchise bought the most production with its
total spend (salary + luxury tax).

v2 is **calibrated against real outcomes** over six seasons (2018-19, 2021-22 →
2025-26; the two COVID-distorted seasons are excluded): the pillar weights are
optimized so that the metric's performance core maximally predicts All-NBA selections
and team win totals, and fitted star/rotation premiums make it rank championship rosters.
Pre-2023 seasons use the 2017-CBA luxury-tax schedule with era-correct repeater status.

---

## 1. Performance side

Stats from stats.nba.com (Regular Season). Z-scores calibrated on the qualified pool
(≥ 500 minutes), applied to everyone; missing inputs get neutral pillars.

| Pillar | Definition | Calibrated weight |
|---|---|---|
| Box production | `0.5·z(per-36 weighted box) + 0.5·z(total weighted box)` | **0.25** |
| Scoring efficiency | `z((TS% − league TS%) × USG%)` | **0.15** |
| Offensive impact | `z([0.6·(E_OFF − lg) + 0.4·(ON_OFF − lg)] × rel)` | **0.15** |
| Defensive impact | `z([0.6·(lg − E_DEF) + 0.4·(lg − ON_DEF)] × rel)` | **0.15** |
| Advanced (PIE) | `z(PIE)` | **0.30** |

`rel = MIN/(MIN+500)` shrinks noisy low-minute ratings. The weighted box composite is
`1.0·PTS + 0.7·REB + 1.4·AST + 2.2·STL + 2.0·BLK − 1.4·TOV`. The 50/50 rate/volume
blend fixes v1's low-minute per-36 bias.

```
CPI = Σ wᵢ·zᵢ            (weights above, from output/calibrated_weights.json)
PS  = 100 · Φ(CPI)                       # 0-100 performance score
A   = sqrt(min(MIN, 2200) / 2200)        # availability
WPS = PS · (0.55 + 0.45·A)               # weighted performance score
```

## 2. Cost side (season-correct CBA)

Per-season constants in `rvalue/config.py` — cap, tax line, aprons, and the correct
luxury-tax schedule for each league year (2017-CBA rates through 2024-25; 2023-CBA
rates with cap-indexed brackets from 2025-26; repeater status from public reporting).

```
Payroll = Σ salaries (dead money included, two-way excluded; server-side team
          aggregates fill any per-player scrape gaps)
TE      = Payroll + luxury tax
EC_i    = salary_i × (TE/Payroll)        # tax gross-up per salary dollar
CostIndex_i = EC_i / mean(TE)            # share of an average team's total budget
```

## 3. R-Value

```
R = 100 · normalize( WPS × (mean(CostIndex)/CostIndex)^0.35 )   # league mean 100
TeamR = 100 · normalize( Σ(WPS·MIN) / TE )                      # production per dollar
TITLE = z(TEAM_PERF) + a·z(TOP2_WPS) + b·z(ROT8_WPS)            # fitted premiums
```

TOP2_WPS is the star ceiling (best two players); ROT8_WPS is the minutes-weighted
WPS of the eight most-played players — the de facto rotation playoff basketball
compresses to. Calibration fits (a, b) so actual champions rank as highly as
possible; currently a=0.2, b=1.8 (fit on six champions) — rotation quality
dominates the star premium, since a team's top two are usually inside its
top-eight rotation anyway.
Teams also get `active_dollar_share`: the share of payroll paid to players with
≥500 minutes (the rest bought bench decoration or dead money).

## 4. Validation (backtest.py, six seasons 2018-19 → 2025-26, ex-COVID)

Predicting All-NBA selections, recall@15 (higher is better):

| Metric | mean |
|---|---|
| **R-Value performance core** | **0.822** |
| PIE × minutes | 0.711 |
| Raw plus-minus | 0.344 |
| Estimated net rating | 0.233 |

Predicting regular-season wins, Spearman: R-Value perf core **0.917** vs team net
rating 0.952 (net rating is point differential — near-definitionally wins — and cannot
rank players or price contracts; R-Value does all three from one framework).
Title score ranked the actual champion #1, #2, #6, #1, #2, #3 across the six
backtest seasons (mean 2.50); the one real miss is the 2022-23 Nuggets, who
coasted to a #9 regular-season performance core before winning it all.

## 5. Prediction (predict.py, 2026-27)

Player CPI projected as a 0.55/0.30/0.15 blend of the last three seasons plus an age
curve; minutes from a two-season blend (raw for individual durability, roster-
normalized for team aggregation); rosters and salaries from contracts signed as of
run date; unsigned FAs and rookies enter at replacement level. Outputs predicted
seeds, title contenders, All-NBA, and team/player R-Values.

## Run it

```
pip install -r requirements.txt
python main.py       # current-season R-Value (cached fetches in data/)
python backtest.py   # 6-season backtest, weight calibration, metric comparison
python crossval.py   # leave-one-season-out validation (the honest accuracy numbers)
python predict.py    # 2026-27 predictions (seeds, champion, All-NBA, value)
```

`backtest.py` writes `output/calibrated_weights.json`, which the other scripts
read, so run it first on a clean checkout.

## Reproducing

`data/` is not committed: it holds cached copies of third-party pages that are
not ours to redistribute. It is rebuilt automatically — the first run of any
script fetches what it needs and caches it, so a clean checkout reproduces every
number in the report from source. Budget ~20 minutes for that first run, which
is dominated by rate-limited (0.6s) salary fetches; every run afterwards is
offline.

Two caveats on reproduction. Stats come from `stats.nba.com` via `nba_api` and
are stable. Salary pages are live documents whose historical rows change when
players re-sign, so a rebuild months from now may recover slightly different
coverage for departed role players; the committed CSVs under `output/` are the
snapshot the report was generated from.

## Validation

Accuracy figures printed by `backtest.py` are **in-sample** — the pillar weights
and title premiums were grid-searched to maximize exactly those quantities on
exactly those seasons. `crossval.py` refits every parameter on N−1 seasons and
scores the held-out one; those are the numbers to cite. Rival metrics
(PIE × minutes, plus-minus, estimated net rating) have no fitted parameters, so
their figures are already out-of-sample and are directly comparable either way.

## Known simplifications
- Repeater-tax status and 2026-27 cap figures are from public reporting/projections.
- Traded players count for the team where they finished; all salary paid counts
  against the paying team.
- Salary history for players no longer under contract is backfilled from player
  pages where possible, hand-patched for notable FAs (`KNOWN_SALARIES`), otherwise
  imputed at an age-proxied minimum and flagged `salary_imputed`.
- 2026-27 rosters are mid-free-agency snapshots; predictions shift as signings land.
