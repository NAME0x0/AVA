"""Figure: Wilson interval width vs sample size for GSM8K, illustrating why a
50-item probe cannot separate 48% from the full-set 35.3%."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

z = 1.96
p = 0.353
n = np.arange(20, 1320)


def wilson(phat, n, z=1.96):
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (center - half) * 100, (center + half) * 100


lo, hi = wilson(p, n)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.fill_between(n, lo, hi, color="#c7d7ee", label="95% Wilson interval")
ax.axhline(35.3, color="#1f4e9c", lw=1.2, label="Full-set value (35.3%)")
ax.plot(1319, 35.3, "s", color="#1f4e9c", ms=5)
ax.plot(50, 48.0, "o", color="#c0392b", ms=6, label="50-item probe (48%)")
ax.set_xscale("log")
ax.set_xlabel("Evaluation sample size $n$")
ax.set_ylabel("GSM8K accuracy (%)")
ax.set_xlim(20, 1319)
ax.set_ylim(15, 60)
ax.set_xticks([20, 50, 100, 200, 500, 1319])
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.legend(fontsize=6, loc="upper right", frameon=False)
ax.grid(True, alpha=0.3, lw=0.4)
fig.tight_layout()
out = Path("D:/AVA/paper/figures/subset_variance.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
