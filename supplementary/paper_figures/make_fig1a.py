# Figure 1(a): per-query latency (ms) split into rotation vs partial scan, ADSampling vs FlipFly.
# Absolute milliseconds with a broken y-axis (XUltra's ADSampling bar reaches 59 ms).
# Left bar of each pair = ADSampling (plain), right bar = FlipFly (hatched, label = speedup).
# Data: src/batch/query_performance/time_table_K100_R95.tex (written by
# print_time_table.py; K=100, recall@100=0.95, ms/query). Output: intro_overview_a.pdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "hatch.linewidth": 0.4,
})

names    = ["Trevi\n4,096", "OpenAI\n3,072", "Arxiv\n4,096", "XUltra\n12,288"]
ads_scan = np.array([1.861, 2.935, 5.869, 7.059])
ads_rot  = np.array([4.636, 1.927, 4.608, 51.899])
sad_scan = np.array([1.718, 2.773, 6.502, 7.711])
sad_rot  = np.array([0.016, 0.016, 0.017, 0.069])
ads_tot, sad_tot = ads_scan + ads_rot, sad_scan + sad_rot
# FlipFly rotation (< 0.07 ms, < 1% of latency) is thinner than a line at this scale.
# Draw it as a thin cap of minimum visible height so readers see that the stage exists.
ROT_MIN_VIS = 0.22  # ms, about 1.2 pt on the lower axis
sad_rot_vis = np.maximum(sad_rot, ROT_MIN_VIS)
c_rot, c_scan, c_ours = "#C9756D", "#DADADA", "#1F3A93"

# Panel aspect must match intro_overview_b.pdf (141.0 x 148.6 pt) so both halves align in the paper.
W = 1.72
TOP, BOT = 0.905, 0.115
fig = plt.figure(figsize=(W, W * 148.6 / 141.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.2], hspace=0.08, left=0.15, right=0.99, top=TOP, bottom=BOT)
ax_hi = fig.add_subplot(gs[0])
ax_lo = fig.add_subplot(gs[1], sharex=ax_hi)
LO, HI = (0, 13.5), (51.0, 62.0)

x = np.arange(4); w = 0.36; gap = 0.06
xa, xs = x - (w + gap) / 2, x + (w + gap) / 2
for ax in (ax_hi, ax_lo):
    ax.bar(xa, ads_scan, w, color=c_scan, edgecolor="k", linewidth=0.4)
    ax.bar(xa, ads_rot, w, bottom=ads_scan, color=c_rot, edgecolor="k", linewidth=0.4)
    ax.bar(xs, sad_scan, w, color=c_scan, edgecolor="k", linewidth=0.4, hatch="////")
    ax.bar(xs, sad_rot_vis, w, bottom=sad_scan, color=c_rot, edgecolor="k", linewidth=0.3)
ax_lo.set_ylim(*LO); ax_hi.set_ylim(*HI)

# labels: total ms above ADSampling, speedup above FlipFly, rotation share inside the red segment
for i in range(4):
    ax_t = ax_hi if ads_tot[i] > LO[1] else ax_lo
    ax_t.text(xa[i], ads_tot[i] + 0.3, f"{ads_tot[i]:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax_lo.text(xs[i], sad_scan[i] + sad_rot_vis[i] + 0.3, f"{ads_tot[i] / sad_tot[i]:.1f}" + r"$\times$", ha="center",
               va="bottom", fontsize=5.8, color=c_ours, fontweight="bold")
    top_vis = min(ads_tot[i], LO[1])
    ax_lo.text(xa[i], (ads_scan[i] + top_vis) / 2, f"{100 * ads_rot[i] / ads_tot[i]:.0f}%",
               ha="center", va="center", fontsize=5.0, color="white", rotation=90)

# axes styling and break marks
for ax in (ax_hi, ax_lo):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1)
ax_hi.spines["bottom"].set_visible(False)
ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax_lo.set_yticks([0, 5, 10]); ax_hi.set_yticks([55, 60])
ax_lo.tick_params(axis="x", length=0, pad=2)
ax_lo.set_xticks(x); ax_lo.set_xticklabels(names, fontsize=6, linespacing=1.0)
ax_lo.set_xlim(-0.55, 3.55)
for dy in (0.0, 0.03):
    ax_lo.plot([-0.035, 0.035], [1.005 + dy - 0.012, 1.005 + dy + 0.012], transform=ax_lo.transAxes,
               color="k", lw=0.5, clip_on=False)
fig.text(0.015, (TOP + BOT) / 2, "latency per query (ms)", rotation=90, ha="left", va="center", fontsize=6)

# legends: methods (longer labels) in the empty upper-left of the top axis,
# components (shorter labels) in the empty upper-left of the bottom axis, clear of the "10.5" label.
# Icons are enlarged and the hatch is denser than on the bars so that the stripes remain visible.
kw = dict(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.5, handleheight=1.1,
          handletextpad=0.35, borderaxespad=0, labelspacing=0.25)
leg = ax_hi.legend(handles=[Patch(facecolor="white", edgecolor="k", linewidth=0.4, label="ADSampling"),
                            Patch(facecolor="white", edgecolor="k", linewidth=0.4, hatch="//////",
                                  label="FlipFly (ours)")],
                   bbox_to_anchor=(0.03, 1.12), **kw)
leg.get_texts()[1].set_color(c_ours)
ax_lo.legend(handles=[Patch(facecolor=c_rot, edgecolor="k", linewidth=0.4, label="rotation"),
                      Patch(facecolor=c_scan, edgecolor="k", linewidth=0.4, label="partial scan")],
             bbox_to_anchor=(0.03, 0.98), **kw)
fig.text(0.57, 0.995, "(a) Where the latency goes", ha="center", va="top", fontsize=7, fontweight="bold")
fig.savefig("intro_overview_a.pdf")
print("wrote intro_overview_a.pdf")
