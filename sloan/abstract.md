# R-Value: Pricing NBA Production Against True Team Expenditure

**Introduction.** Public player-value metrics measure production but ignore what
that production costs. Nominal salary understates cost badly: under the luxury
tax, a dollar paid by a repeater taxpayer can cost its franchise several times a
dollar paid by a team under the line. We ask which players and teams deliver the
most production per dollar of *true* expenditure, and whether cost-efficiency
predicts championships. The question is timely because the 2023 CBA's apron
system was designed explicitly to constrain high-spending rosters.

**Methods.** R-Value pairs a performance core with a CBA-correct cost model. The
core blends five z-scored pillars — box production (rate and volume),
usage-scaled shooting efficiency, minute-shrunk on-court offensive and defensive
impact, and PIE — into a composite index, maps it through the normal CDF to a
0–100 score, and discounts it for availability. Pillar weights are fit by
two-stage simplex grid search maximizing All-NBA recall@15 and team-win rank
correlation. The cost model computes each season's luxury tax from the bracket
schedule then in force (2017 CBA through 2022-23, 2023 CBA after), applies
repeater rates, and grosses up every salary by its team's total-expenditure
multiplier. Player R-Value divides availability-weighted production by a
dampened cost share; team R-Value divides minute-weighted production by total
expenditure. Data covers six seasons — 2018-19 and 2021-22 through 2025-26 —
excluding the COVID-disrupted 2019-20 and 2020-21, whose uneven schedules and
pro-rated tax settlements break the availability and cost models respectively.

**Results.** Under leave-one-season-out cross-validation, refitting every
parameter without the season it is scored on, the performance core recovers
**78.9%** of All-NBA selections in its top 15, against 71.1% for PIE×minutes,
34.4% for raw plus-minus and 23.3% for estimated net rating, and correlates with
team wins at **ρ = 0.914**. Weights are stable: four of six folds select an
identical vector, and the offensive-impact weight is identical in all six.

Championships are rarely bought efficiently (Figure 1). Across six champions,
production-per-dollar ranked 16th, 29th, 14th, 13th, 3rd and 14th of 30. The
2021-22 Warriors won at an R-Value of 49 against a league average of 100, on a
$179M payroll carrying a $191M repeater tax. A fitted title score weights
eight-man rotation quality roughly nine times the top-two star premium, and
ranks actual champions **2.83rd of 30** out of sample.

**Conclusion.** Cost-efficiency and contention are near-orthogonal: teams
convert brief value windows into titles by spending through them, and the
league's most efficient rosters are typically two years from contention rather
than in it. R-Value quantifies that trade-off in CBA-correct dollars, letting
front offices price roster decisions against real expenditure rather than
nominal salary, and ranks players, rates teams and forecasts seasons from a
single model.

Code and data: [GitHub repository]

---

### Figure 1
`figure1_champions_value.png` — Team R-Value for all 30 teams across six
seasons, champion highlighted. Five of six champions sit at or below the league
average of 100.

### Table 1 — Leave-one-season-out cross-validation

Every parameter (five pillar weights, two title premiums) is refit on the other
five seasons, then scored on the held-out season.

| Held-out season | R-Value recall@15 | rho with wins | Champion title rank |
|---|---|---|---|
| 2018-19 | 0.800 | 0.935 | 1 |
| 2021-22 | 0.800 | 0.882 | 3 |
| 2022-23 | 0.733 | 0.880 | 6 |
| 2023-24 | 0.800 | 0.926 | 1 |
| 2024-25 | 0.800 | 0.932 | 3 |
| 2025-26 | 0.800 | 0.927 | 3 |
| **Mean (out-of-sample)** | **0.789** | **0.914** | **2.83 of 30** |

Benchmarks, which carry no fitted parameters: PIE x minutes 0.711,
raw plus-minus 0.344, estimated net rating 0.233.
