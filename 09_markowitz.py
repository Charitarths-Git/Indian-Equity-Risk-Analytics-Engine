"""
TIER-1 (3) -- Markowitz mean-variance optimization + efficient frontier.

Ties RISK to ALLOCATION: how much risk does the cap-weighted portfolio leave on
the table vs the minimum-variance and maximum-Sharpe portfolios?

  * Min-variance  : lowest possible volatility (long-only).
  * Max-Sharpe    : best return-per-unit-risk (tangency portfolio).
  * Efficient frontier: the best achievable return for each risk level.

Annualised: mu*252, Sigma*252. Risk-free assumed 6.5% (Indian ~repo/G-sec).

Run:  python 09_markowitz.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

import config as C
import utils

PV = C.PORTFOLIO_VALUE_INR
RF = 0.065
port, assets, wdf = utils.load()
tickers = list(assets.columns)
w_cap = wdf.loc[tickers, "weight"].values
mu = assets.mean().values * 252
S = assets.cov().values * 252
n = len(mu)
z99 = -norm.ppf(0.01)


def pstats(wv):
    r = float(wv @ mu); v = float(np.sqrt(wv @ S @ wv))
    return r, v, (r - RF) / v


def var99_1day(v_annual):
    return z99 * (v_annual / np.sqrt(252)) * PV       # Gaussian 1-day 99% VaR (Rs)


cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
bnds = [(0.0, 1.0)] * n
x0 = np.ones(n) / n

w_mv = minimize(lambda w: np.sqrt(w @ S @ w), x0, bounds=bnds,
                constraints=cons, method="SLSQP").x
w_ms = minimize(lambda w: -((w @ mu - RF) / np.sqrt(w @ S @ w)), x0, bounds=bnds,
                constraints=cons, method="SLSQP").x

print("=" * 78)
print("MARKOWITZ OPTIMIZATION  (annualised; risk-free 6.5%)")
print("=" * 78)
print(f"{'portfolio':<18}{'return':>9}{'vol':>9}{'Sharpe':>9}{'1-day 99% VaR':>18}")
for name, wv in [("Cap-weighted (now)", w_cap),
                 ("Min-variance", w_mv),
                 ("Max-Sharpe", w_ms)]:
    r, v, sh = pstats(wv)
    print(f"{name:<18}{r*100:8.2f}%{v*100:8.2f}%{sh:9.2f}"
          f"{utils.inr(var99_1day(v)):>18}")

r_cap, v_cap, _ = pstats(w_cap)
r_mv, v_mv, _ = pstats(w_mv)
var_cut = 1 - var99_1day(v_mv) / var99_1day(v_cap)
print(f"\n=> Min-variance cuts 1-day 99% VaR by {var_cut*100:.1f}% "
      f"({utils.inr(var99_1day(v_cap))} -> {utils.inr(var99_1day(v_mv))}) "
      f"vs the cap-weighted book.")

# top holdings of optimized portfolios
def top_holdings(wv, k=5):
    s = pd.Series(wv, index=[t.replace(".NS", "") for t in tickers])
    return ", ".join(f"{t} {x*100:.0f}%" for t, x in s.sort_values(ascending=False).head(k).items())

print(f"\nMin-variance tilts to : {top_holdings(w_mv)}")
print(f"Max-Sharpe tilts to   : {top_holdings(w_ms)}")

# save weights
pd.DataFrame({"cap_weight": w_cap, "min_var": w_mv, "max_sharpe": w_ms},
             index=tickers).round(4).to_csv(
    os.path.join(C.OUT_DIR, "optimized_weights.csv"))

# ---- efficient frontier (long-only) ----
targets = np.linspace(mu.min(), mu.max(), 40)
fr_v, fr_r = [], []
for t in targets:
    c = ({"type": "eq", "fun": lambda w: w.sum() - 1},
         {"type": "eq", "fun": lambda w, t=t: w @ mu - t})
    res = minimize(lambda w: np.sqrt(w @ S @ w), x0, bounds=bnds,
                   constraints=c, method="SLSQP")
    if res.success:
        fr_v.append(float(np.sqrt(res.x @ S @ res.x))); fr_r.append(t)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.array(fr_v) * 100, np.array(fr_r) * 100, "b-", lw=1.5, label="efficient frontier")
ax.scatter(np.sqrt(np.diag(S)) * 100, mu * 100, c="gray", s=18, label="individual stocks")
for name, wv, col in [("Cap-weighted", w_cap, "black"),
                      ("Min-variance", w_mv, "green"),
                      ("Max-Sharpe", w_ms, "red")]:
    r, v, _ = pstats(wv)
    ax.scatter(v * 100, r * 100, c=col, s=90, marker="*", zorder=5, label=name)
ax.set_xlabel("annualised volatility (%)"); ax.set_ylabel("annualised return (%)")
ax.set_title("Efficient frontier -- Indian 15-stock portfolio"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(C.OUT_DIR, "efficient_frontier.png"), dpi=130)
plt.close(fig)
print("\nSaved: outputs/optimized_weights.csv, outputs/efficient_frontier.png")
