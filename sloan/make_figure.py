"""Figure 1 for the SSAC27 abstract: champions sit in the inefficient half
of the league's production-per-dollar distribution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a988f"
FIELD, CHAMP = "#c9c7bf", "#2a78d6"
SEASONS = ["2018-19", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

fig, ax = plt.subplots(figsize=(9.6, 5.0), facecolor=SURF)
ax.set_facecolor(SURF)
rng = np.random.default_rng(7)

for i, s in enumerate(SEASONS):
    t = pd.read_csv(f"output/team_rvalue_{s}.csv")
    t["rank"] = t.TEAM_R_VALUE.rank(ascending=False)
    ch = t[t.champion].iloc[0]
    x = i + rng.uniform(-0.17, 0.17, len(t))
    ax.scatter(x, t.TEAM_R_VALUE, s=26, color=FIELD, zorder=2,
               edgecolor=SURF, linewidth=0.6)
    ax.scatter([i], [ch.TEAM_R_VALUE], s=132, color=CHAMP, zorder=4,
               edgecolor=SURF, linewidth=1.6)
    ax.annotate(f"{ch.TEAM_ABBREVIATION}\n#{int(ch['rank'])} of 30",
                xy=(i, ch.TEAM_R_VALUE), xytext=(0, -16),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.6, color=INK, linespacing=1.35)

ax.axhline(100, color=INK, lw=1.1, ls=(0, (5, 4)), alpha=0.6, zorder=1)


ax.set_xticks(range(len(SEASONS)))
ax.set_xticklabels(SEASONS, fontsize=9.6, color=INK2)
ax.set_xlim(-0.55, 5.55)
ax.set_ylim(30, None)
ax.set_ylabel("Team R-Value  (production per dollar)",
              fontsize=9.4, color=INK2, labelpad=9)
ax.tick_params(axis="y", colors=INK2, labelsize=9, length=3, width=0.8)
ax.tick_params(axis="x", length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d8d6d0")
ax.grid(axis="y", color="#ebe9e3", lw=0.7)
ax.set_axisbelow(True)

from matplotlib.lines import Line2D
ax.scatter([], [], s=26, color=FIELD, label="the other 29 teams")
ax.scatter([], [], s=90, color=CHAMP, label="NBA champion")
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color=INK, lw=1.1, ls=(0, (5, 4)), alpha=0.6))
labels.append("league average = 100")
leg = ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper left",
                bbox_to_anchor=(0.0, 1.13), ncol=3, handletextpad=0.6,
                columnspacing=2.2)
for txt in leg.get_texts():
    txt.set_color(INK2)

fig.subplots_adjust(top=0.88, bottom=0.14, left=0.085, right=0.975)
fig.savefig("sloan/figure1_champions_value.png", dpi=300, facecolor=SURF)
print("saved sloan/figure1_champions_value.png")

for s in SEASONS:
    t = pd.read_csv(f"output/team_rvalue_{s}.csv")
    ch = t[t.champion].iloc[0]
    print(f"  {s}  {ch.TEAM_ABBREVIATION}  R={ch.TEAM_R_VALUE:6.1f}  "
          f"rank {int(t.TEAM_R_VALUE.rank(ascending=False)[ch.name]):2d}/30")
