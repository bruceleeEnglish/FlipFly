"""Regenerate rotation_only_K100.pdf as a compact 1x4 strip.

Values are the rot_ms column of supplementary/qps_table/qps_table.txt at K=100, i.e.
exactly the numbers the previous 2x2 version annotated.  Nothing is recomputed
here; the layout is the only thing that changes.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

OPS = ['ADSampling', 'DADE', 'DDCres', 'FlipFly', 'FlipFly-D']
COL = ['#4C72B0', '#55A868', '#937860', '#C44E52', '#2B2B2B']

# (panel title, rot_ms per operator) -- qps_table.txt, K=100
PANELS = [
    ('(a) Trevi',      [4.636,  4.617,  4.810,  0.016, 0.018]),
    ('(b) Arxiv',      [4.608,  5.610,  4.924,  0.017, 0.020]),
    ('(c) XUltra',     [51.899, 53.095, 51.861, 0.069, 0.075]),
    ('(d) OpenAI3072', [1.927,  2.150,  2.343,  0.016, 0.020]),
]

plt.rcParams.update({
    'font.size': 5.5,
    'axes.labelsize': 5.5,
    'axes.titlesize': 6.0,
    'xtick.labelsize': 5.0,
    'ytick.labelsize': 5.0,
    'legend.fontsize': 5.5,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
})

fig, axes = plt.subplots(1, 4, figsize=(3.46, 1.26))
x = range(len(OPS))

for ax, (title, vals) in zip(axes, PANELS):
    ax.bar(x, vals, color=COL, edgecolor='black', linewidth=0.4, width=0.70)
    ax.set_yscale('log')
    ax.set_title(title, pad=0.8)
    ax.set_xticks([])
    ax.set_xlim(-0.66, len(OPS) - 0.34)
    # Per-panel range, with the headroom sized to the rotated value labels:
    # on a shared range the three low corpora sit a decade below XUltra and
    # their titles drift away from their bars.
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo / 2.0, hi * 24.0)
    for xi, v in zip(x, vals):
        ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                    xytext=(0, 1.2), ha='center', va='bottom',
                    rotation=90, fontsize=4.8)
    # Two decades per label, no minor ticks: the bars span four to five
    # decades, so a tick on every one is unreadable at this size.
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='y', length=2.0, width=0.4, pad=0.8)
    # No frame at all: the baseline the bars stand on is the only rule the
    # plot needs, and the y ticks anchor their own labels.
    for name, s in ax.spines.items():
        s.set_visible(name == 'bottom')
    ax.spines['bottom'].set_linewidth(0.5)

axes[0].set_ylabel('rotation (ms)', labelpad=0.5)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black',
                         linewidth=0.4) for c in COL]
fig.legend(handles, OPS, loc='upper center', ncol=len(OPS),
           frameon=False, bbox_to_anchor=(0.5, 1.012),
           handlelength=0.9, handletextpad=0.3, columnspacing=0.8)

fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=0.45)
fig.savefig('rotation_only_K100.pdf', bbox_inches='tight', pad_inches=0.005)
print('wrote rotation_only_K100.pdf')
