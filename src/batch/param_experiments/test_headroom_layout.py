"""Quick local test: regenerate headroom.pdf from the data already
stored in headroom_table.tex / proxy_agreement.tex, without
re-streaming the remote SRHT files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_headroom import plot

# Hard-coded from output/headroom_table.tex (most recent run).
# Columns there are: alpha, obs(0.0625), pred(0.0625), obs(0.125),
# pred(0.125), obs(0.25), pred(0.25).
RAW = {
    "trevi":      (0.508, 0.250, 0.057, 0.303, 0.117, 0.389, 0.240),
    "OpenAI3072": (0.075, 0.844, 0.799, 0.869, 0.845, 0.898, 0.894),
    "arxiv-for-fanns-large":
                  (0.166, 0.656, 0.575, 0.711, 0.660, 0.777, 0.758),
    "XUltra":     (0.064, 0.861, 0.827, 0.884, 0.867, 0.911, 0.910),
}

RATIOS = [1 / 16, 1 / 8, 1 / 4]

results = {}
for ds, row in RAW.items():
    alpha, o1, p1, o2, p2, o3, p3 = row
    results[ds] = {
        "alpha": alpha,
        "observed":  {RATIOS[0]: o1, RATIOS[1]: o2, RATIOS[2]: o3},
        "predicted": {RATIOS[0]: p1, RATIOS[1]: p2, RATIOS[2]: p3},
    }

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "output", "headroom.pdf")
plot(results, RATIOS, out)
print("done")
