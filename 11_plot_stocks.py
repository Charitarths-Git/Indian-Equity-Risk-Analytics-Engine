"""
STAGE 11 -- Visualise all 15 stocks over time.

Two views (saved to outputs/):
  stocks_rebased.png : all 15 rebased to Rs 100 on a LOG axis -> percentage moves
                       are visually comparable (who grew, who's volatile, COVID dip).
  stocks_grid.png    : 3x5 small multiples -> each stock's own price fluctuation.

Run:  python 11_plot_stocks.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

prices = pd.read_csv(C.PRICES_CSV, index_col=0, parse_dates=True)
names = [c.replace(".NS", "") for c in prices.columns]
COVID = (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))

# ---- (1) all rebased to 100 (log scale) ----
reb = prices / prices.iloc[0] * 100
fig, ax = plt.subplots(figsize=(13, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
for i, c in enumerate(prices.columns):
    ax.plot(reb.index, reb[c], lw=1.0, color=colors[i], label=names[i])
ax.axvspan(*COVID, color="red", alpha=0.08, label="COVID crash")
ax.set_yscale("log")
ax.set_title("Growth of Rs 100 invested on 2018-01-01 (dividend/split-adjusted, log scale)")
ax.set_ylabel("value of Rs 100 (log)")
ax.legend(ncol=4, fontsize=7, loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(C.OUT_DIR, "stocks_rebased.png"), dpi=130)
plt.close(fig)

# ---- (2) small multiples ----
fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True)
for ax, c, nm in zip(axes.flat, prices.columns, names):
    ax.plot(prices.index, prices[c], lw=0.8, color="steelblue")
    ax.axvspan(*COVID, color="red", alpha=0.10)
    ax.set_title(nm, fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6)
fig.suptitle("Individual adjusted-price history (Rs) -- each of the 15 stocks", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(C.OUT_DIR, "stocks_grid.png"), dpi=130)
plt.close(fig)

# ---- printed summary: total return over the window ----
tot = (prices.iloc[-1] / prices.iloc[0] - 1).sort_values(ascending=False)
ann = (1 + tot) ** (252 / len(prices)) - 1
print("Total return 2018-01-01 -> 2026-06-08 (adjusted):")
for c in tot.index:
    print(f"  {c.replace('.NS',''):<13} {tot[c]*100:+8.1f}%   (CAGR {ann[c]*100:+5.1f}%)")
print(f"\n  Best : {tot.index[0].replace('.NS','')} ({tot.iloc[0]*100:+.0f}%)")
print(f"  Worst: {tot.index[-1].replace('.NS','')} ({tot.iloc[-1]*100:+.0f}%)")
print("\nSaved: outputs/stocks_rebased.png, outputs/stocks_grid.png")
