"""
TIER-1 (4) -- Bootstrap confidence intervals on VaR / CVaR.

Every VaR number is a point estimate from a finite sample, so it has sampling
error. A BLOCK bootstrap resamples blocks of consecutive returns (preserving
autocorrelation & volatility clustering -- a plain i.i.d. bootstrap would
destroy them), recomputes the risk measure many times, and reports the spread.

Output: "99% 1-day VaR = Rs X  (95% CI [Rs lo, Rs hi])".

Run:  python 10_bootstrap_ci.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import utils

PV = C.PORTFOLIO_VALUE_INR
port, _, _ = utils.load()
r = port.values
N = len(r)
L = 10          # block length (~2 weeks) to retain dependence
B = 2000        # bootstrap replications
rng = np.random.default_rng(C.MC_SEED)


def block_sample(r, L, size):
    n_blocks = int(np.ceil(size / L))
    starts = rng.integers(0, len(r) - L, n_blocks)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:size]
    return r[idx]


def var_cvar(x, a):
    q = np.quantile(x, a)
    return -q, -x[x <= q].mean()


print("=" * 74)
print(f"BOOTSTRAP CONFIDENCE INTERVALS  (block L={L}, B={B} resamples)")
print("=" * 74)
print(f"{'conf':>6}{'measure':>9}{'point est':>16}{'95% CI':>30}")

rows = []
boot_store = {}
for cl in C.CONFIDENCE_LEVELS:
    a = 1 - cl
    v_pt, c_pt = var_cvar(r, a)
    vboot = np.empty(B); cboot = np.empty(B)
    for b in range(B):
        rb = block_sample(r, L, N)
        vboot[b], cboot[b] = var_cvar(rb, a)
    boot_store[cl] = (vboot, cboot)
    for label, pt, dist in [("VaR", v_pt, vboot), ("CVaR", c_pt, cboot)]:
        lo, hi = np.percentile(dist, [2.5, 97.5])
        rows.append({"confidence": f"{int(cl*100)}%", "measure": label,
                     "point_inr": round(pt * PV), "ci_low_inr": round(lo * PV),
                     "ci_high_inr": round(hi * PV),
                     "ci_width_%": round((hi - lo) / pt * 100, 1)})
        print(f"{int(cl*100):>5}%{label:>9}{utils.inr(pt*PV):>16}"
              f"   [{utils.inr(lo*PV)}, {utils.inr(hi*PV)}]")

pd.DataFrame(rows).to_csv(os.path.join(C.OUT_DIR, "bootstrap_ci.csv"), index=False)

# chart: bootstrap distribution of 99% VaR & CVaR
v99, c99 = boot_store[0.99]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(v99 * PV / 1e5, bins=50, alpha=0.6, color="teal", label="99% VaR")
ax.hist(c99 * PV / 1e5, bins=50, alpha=0.6, color="crimson", label="99% CVaR")
for d, col in [(v99, "teal"), (c99, "crimson")]:
    lo, hi = np.percentile(d * PV / 1e5, [2.5, 97.5])
    ax.axvline(lo, color=col, ls="--", lw=1); ax.axvline(hi, color=col, ls="--", lw=1)
ax.set_xlabel("Rs lakh"); ax.set_title("Bootstrap sampling distribution of 99% VaR / CVaR "
                                       "(dashed = 95% CI)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(C.OUT_DIR, "bootstrap_ci.png"), dpi=130)
plt.close(fig)

print("\nInterpretation: the CI WIDTH is how much your risk estimate could be off")
print("purely from sample noise -- wider for the deeper (99%) tail, as expected.")
print("Saved: outputs/bootstrap_ci.csv, outputs/bootstrap_ci.png")
