"""Calibrate R-Value weights against real outcomes (All-NBA, wins, titles).

Objective, averaged over history seasons:
  0.5 * recall@15  (top-15 players by WPS vs actual All-NBA selections)
+ 0.5 * spearman   (team performance core vs regular-season wins)

A star-premium lambda is then fitted so the title score
  TITLE = z(TEAM_PERF) + lambda * z(TOP2_WPS)
ranks actual champions as highly as possible.
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

from . import config, model


def _zs(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


def recall_at_15(players: pd.DataFrame, season: str, score_col: str) -> float:
    truth = {model.norm_name(n) for n in config.ALL_NBA[season]}
    pool = players[players["ELIGIBLE"]].copy()
    pool["nn"] = pool["PLAYER_NAME"].map(model.norm_name)
    top = set(pool.nlargest(15, score_col)["nn"])
    return len(top & truth) / 15.0


def wps_from_weights(players: pd.DataFrame, w: np.ndarray) -> pd.Series:
    cpi = (w[0] * players["z_box"] + w[1] * players["z_eff"] + w[2] * players["z_off"]
           + w[3] * players["z_dfn"] + w[4] * players["z_adv"])
    ps = 100.0 * norm.cdf(cpi)
    return ps * (config.AVAIL_FLOOR + (1 - config.AVAIL_FLOOR) * players["AVAIL"])


def team_perf_from_wps(players: pd.DataFrame, wps: pd.Series) -> pd.DataFrame:
    t = players[["TEAM_ABBREVIATION", "MIN"]].copy()
    t["wm"] = wps * t["MIN"]
    g = t.groupby("TEAM_ABBREVIATION").agg(wm=("wm", "sum"), m=("MIN", "sum"))
    return (g["wm"] / g["m"]).rename("perf")


def _grid(ticks: np.ndarray, floor: float):
    for combo in itertools.product(ticks, repeat=4):
        last = 1.0 - sum(combo)
        if last >= floor - 1e-9:
            yield np.array([*combo, last])


WEIGHT_KEYS = ["box", "eff", "off", "dfn", "adv"]


def _search_weights(evals: list) -> tuple[np.ndarray, float]:
    """Two-stage simplex grid search over the five pillar weights.

    Maximizes 0.5*mean(recall@15) + 0.5*mean(spearman with wins) over the
    seasons in `evals` -- stage 1 a coarse sweep, stage 2 a local refinement
    around the stage-1 winner.
    """
    def objective(w: np.ndarray) -> float:
        rec, cor = zip(*(e.evaluate(w) for e in evals))
        return 0.5 * float(np.mean(rec)) + 0.5 * float(np.mean(cor))

    best, best_obj = None, -1.0
    for w in _grid(np.arange(0.05, 1.0, 0.10), 0.05):
        obj = objective(w)
        if obj > best_obj:
            best_obj, best = obj, w
    fine = np.arange(0.05, 1.0, 0.05)
    lo = np.maximum(best[:4] - 0.10, 0.05)
    hi = best[:4] + 0.10
    ticks = [fine[(fine >= lo[i] - 1e-9) & (fine <= hi[i] + 1e-9)] for i in range(4)]
    for combo in itertools.product(*ticks):
        last = 1.0 - sum(combo)
        if last >= 0.05 - 1e-9:
            w = np.array([*combo, last])
            obj = objective(w)
            if obj > best_obj:
                best_obj, best = obj, w
    return best, best_obj


def _title_frames(season_players: dict[str, pd.DataFrame],
                  season_teams: dict[str, pd.DataFrame],
                  seasons: list[str], weights: dict) -> dict[str, pd.DataFrame]:
    """Per-season team frames (perf, top-2, rotation-8) under given weights."""
    frames = {}
    for s in seasons:
        p = model.apply_weights(season_players[s], weights)
        frames[s] = season_teams[s][["TEAM_ABBREVIATION", "champion"]].merge(
            model.team_performance(p), on="TEAM_ABBREVIATION")
    return frames


def _champ_rank(t: pd.DataFrame, a: float, b: float) -> float:
    title = _zs(t["TEAM_PERF"]) + a * _zs(t["TOP2_WPS"]) + b * _zs(t["ROT8_WPS"])
    return float(title.rank(ascending=False)[t["champion"]].iloc[0])


def _fit_title(frames: dict[str, pd.DataFrame],
               seasons: list[str]) -> tuple[float, float, float]:
    """Fit (star, rotation) premiums minimizing champions' mean title rank."""
    best_a, best_b, best_rank = 0.0, 0.0, 99.0
    for a in np.arange(0.0, 2.51, 0.1):
        for b in np.arange(0.0, 2.51, 0.1):
            mean_rank = float(np.mean([_champ_rank(frames[s], a, b) for s in seasons]))
            # strict improvement keeps the smaller premiums on ties
            if mean_rank < best_rank - 1e-9:
                best_rank, best_a, best_b = mean_rank, float(a), float(b)
    return best_a, best_b, best_rank


class _SeasonEval:
    """Precomputed fast evaluator for one season."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame, season: str):
        self.Z = players[["z_box", "z_eff", "z_off", "z_dfn", "z_adv"]].to_numpy()
        self.avail_factor = (config.AVAIL_FLOOR
                             + (1 - config.AVAIL_FLOOR) * players["AVAIL"].to_numpy())
        self.eligible = players["ELIGIBLE"].to_numpy()
        truth = {model.norm_name(n) for n in config.ALL_NBA[season]}
        self.is_all_nba = players["PLAYER_NAME"].map(
            lambda n: model.norm_name(n) in truth).to_numpy()
        # team minute-share matrix (teams x players)
        team_codes, team_idx = pd.factorize(players["TEAM_ABBREVIATION"])
        m = players["MIN"].to_numpy()
        W = np.zeros((len(team_idx), len(players)))
        W[team_codes, np.arange(len(players))] = m
        W /= W.sum(axis=1, keepdims=True)
        self.W = W
        wins = teams.set_index("TEAM_ABBREVIATION")["WINS"]
        self.win_ranks = wins.reindex(team_idx).rank().to_numpy()

    def wps(self, w: np.ndarray) -> np.ndarray:
        return 100.0 * norm.cdf(self.Z @ w) * self.avail_factor

    def evaluate(self, w: np.ndarray) -> tuple[float, float]:
        wps = self.wps(w)
        elig = np.where(self.eligible)[0]
        top15 = elig[np.argpartition(-wps[elig], 15)[:15]]
        recall = self.is_all_nba[top15].sum() / 15.0
        perf = self.W @ wps
        ok = ~np.isnan(self.win_ranks)
        pr = pd.Series(perf[ok]).rank().to_numpy()
        rho = np.corrcoef(pr, self.win_ranks[ok])[0, 1]
        return recall, rho


def calibrate(season_players: dict[str, pd.DataFrame],
              season_teams: dict[str, pd.DataFrame]) -> dict:
    seasons = list(season_players)
    evals = [_SeasonEval(season_players[s], season_teams[s], s) for s in seasons]

    best, best_obj = _search_weights(evals)
    weights = dict(zip(WEIGHT_KEYS, (float(x) for x in best)))

    # fit title-score premiums on actual champions:
    #   TITLE = z(TEAM_PERF) + a*z(TOP2_WPS) + b*z(ROT8_WPS)
    # a rewards star ceiling, b rewards the quality of the de facto
    # 8-man rotation that playoff basketball compresses to
    season_frames = _title_frames(season_players, season_teams, seasons, weights)
    best_a, best_b, best_rank = _fit_title(season_frames, seasons)

    result = {
        "weights": weights,
        "star_lambda": best_a,
        "rot_lambda": best_b,
        "objective": best_obj,
        "champion_mean_title_rank": best_rank,
        "seasons": seasons,
    }
    config.WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    config.WEIGHTS_FILE.write_text(json.dumps(result, indent=1))
    return result


def cross_validate(season_players: dict[str, pd.DataFrame],
                   season_teams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Leave-one-season-out validation of every fitted parameter.

    For each held-out season: refit the pillar weights and the title premiums
    on the other seasons only, then score the held-out season with them. This
    is the honest accuracy estimate -- `calibrate` reports numbers on the same
    seasons it optimized against.

    The rival metrics in `metric_comparison` carry no fitted parameters, so
    their in-sample figures are already out-of-sample and need no CV.
    """
    seasons = list(season_players)
    evals = {s: _SeasonEval(season_players[s], season_teams[s], s) for s in seasons}

    rows = []
    for held in seasons:
        train = [s for s in seasons if s != held]
        w, _ = _search_weights([evals[s] for s in train])
        weights = dict(zip(WEIGHT_KEYS, (float(x) for x in w)))
        recall, rho = evals[held].evaluate(w)
        # the fold's weights change team performance too, so rebuild frames
        frames = _title_frames(season_players, season_teams, seasons, weights)
        a, b, _ = _fit_title(frames, train)
        rows.append({"held_out": held, "recall@15": recall, "spearman_wins": rho,
                     "champ_title_rank": _champ_rank(frames[held], a, b),
                     "star_lambda": a, "rot_lambda": b,
                     **{f"w_{k}": v for k, v in weights.items()}})
        print(f"  fold {held}: recall={recall:.3f} spearman={rho:.3f} "
              f"champ_rank={rows[-1]['champ_title_rank']:.0f}")
    return pd.DataFrame(rows)


def metric_comparison(season_players: dict[str, pd.DataFrame],
                      season_teams: dict[str, pd.DataFrame],
                      weights: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R-Value performance core vs rival metrics on the same ground truth."""
    seasons = list(season_players)
    w = np.array([weights[k] for k in ["box", "eff", "off", "dfn", "adv"]])

    # players: recall@15 vs All-NBA
    player_rows = []
    rivals = {"PIE (total)": None, "E_NET_RATING": "E_NET_RATING",
              "PLUS_MINUS": "PLUS_MINUS"}
    for s in seasons:
        p = season_players[s].copy()
        p["_rv"] = wps_from_weights(p, w)
        p["_pie_total"] = p["PIE"] * p["MIN"]
        scores = {"R-Value (WPS core)": "_rv", "PIE x minutes": "_pie_total",
                  "Est. net rating": "E_NET_RATING", "Raw plus-minus": "PLUS_MINUS"}
        for label, col in scores.items():
            player_rows.append({"season": s, "metric": label,
                                "recall@15": recall_at_15(p, s, col)})
    player_cmp = (pd.DataFrame(player_rows)
                  .pivot(index="metric", columns="season", values="recall@15"))
    player_cmp["mean"] = player_cmp.mean(axis=1)

    # teams: spearman with wins
    team_rows = []
    for s in seasons:
        t = season_teams[s]
        p = season_players[s]
        wps = wps_from_weights(p, w)
        perf = team_perf_from_wps(p, wps).rename("perf_c")
        t = t.merge(perf, left_on="TEAM_ABBREVIATION", right_index=True)
        for label, col in {"R-Value perf core": "perf_c", "Team net rating": "NET_RATING",
                           "Payroll": "payroll", "Team R-Value (value)": "TEAM_R_VALUE"}.items():
            team_rows.append({"season": s, "metric": label,
                              "spearman_wins": spearmanr(t[col], t["WINS"]).statistic})
    team_cmp = (pd.DataFrame(team_rows)
                .pivot(index="metric", columns="season", values="spearman_wins"))
    team_cmp["mean"] = team_cmp.mean(axis=1)
    return player_cmp, team_cmp
