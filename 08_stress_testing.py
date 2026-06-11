"""
TIER-1 (2) -- Stress testing & scenario analysis.

Static VaR describes normal times; stress tests answer "what happens in the
NEXT crisis." Two kinds:

  (A) HISTORICAL REPLAY -- apply an actual crisis window's asset moves to
      TODAY's portfolio: "if this portfolio had lived through COVID, it loses X".
  (B) HYPOTHETICAL SHOCKS -- expert-defined instantaneous shocks by sector
      (market crash, banking crisis, rates +200bp, INR -5%, global risk-off).

Note: data starts 2018, so 2008-GFC cannot be replayed (would need older data);
we replay the worst windows actually in the sample (COVID-2020, 2022 selloff).

Run:  python 08_stress_testing.py
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
port, assets, wdf = utils.load()
tickers = list(assets.columns)
w = wdf.loc[tickers, "weight"].values
sector = dict(zip(tickers, wdf.loc[tickers, "sector"].values))


# ---------- (A) historical replay ----------
def replay(start, end):
    sub = assets.loc[start:end]
    asset_cum = (1 + sub).prod() - 1          # per-asset cumulative return
    return float(asset_cum.values @ w), len(sub)


print("=" * 74)
print("STRESS TEST (A) -- HISTORICAL CRISIS REPLAY (today's weights)")
print("=" * 74)
hist_windows = {
    "COVID crash (19 Feb-23 Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate-hike selloff (Jan-Jun 2022)": ("2022-01-01", "2022-06-17"),
    "Late-2024/25 correction (Oct24-Feb25)": ("2024-10-01", "2025-02-28"),
}
rows = []
for name, (s, e) in hist_windows.items():
    r, ndays = replay(s, e)
    rows.append((name, r, r * PV, ndays))
    print(f"  {name:<42} {r*100:7.2f}%   {utils.inr(r*PV):>14}   ({ndays}d)")

# empirical worst rolling windows for the CURRENT portfolio series
cum = (1 + port).cumprod()
dd = (cum / cum.cummax() - 1)
print(f"\n  Max peak-to-trough drawdown in sample : {dd.min()*100:6.2f}%  "
      f"({utils.inr(dd.min()*PV)})  trough {dd.idxmin().date()}")
for h, lab in [(5, "worst 1-week"), (21, "worst 1-month")]:
    roll = port.rolling(h).sum()
    print(f"  {lab:<22}: {roll.min()*100:6.2f}%  ({utils.inr(roll.min()*PV)})  "
          f"ending {roll.idxmin().date()}")


# ---------- (B) hypothetical sector shocks ----------
def shock_vector(spec, default):
    return np.array([spec.get(sector[t], default) for t in tickers])


scenarios = {
    "Broad market crash (-10% all)":      (shock_vector({}, -0.10)),
    "Banking crisis (banks -20%)":        (shock_vector({"Banking": -0.20}, -0.03)),
    "Rates +200bp (rate-sensitives hit)": (shock_vector(
        {"Banking": -0.05, "Auto": -0.06, "Metals": -0.04, "IT": -0.01}, -0.02)),
    "INR -5% depreciation (IT gains)":    (shock_vector(
        {"IT": 0.04, "Energy": -0.02}, -0.015)),
    "Global risk-off (cyclicals worst)":  (shock_vector(
        {"Metals": -0.12, "IT": -0.05}, -0.07)),
}

print("\n" + "=" * 74)
print("STRESS TEST (B) -- HYPOTHETICAL SECTOR SHOCKS (instantaneous P&L)")
print("=" * 74)
brows = []
for name, shock in scenarios.items():
    pl = float(w @ shock)
    brows.append((name, pl, pl * PV))
    print(f"  {name:<38} {pl*100:7.2f}%   {utils.inr(pl*PV):>14}")

# save + chart
out = pd.DataFrame(
    [(n, r, v) for n, r, v, _ in rows] + [(n, r, v) for n, r, v in brows],
    columns=["scenario", "portfolio_return", "pnl_inr"])
out.to_csv(os.path.join(C.OUT_DIR, "stress_test_results.csv"), index=False)

labels = [r[0] for r in rows] + [b[0] for b in brows]
vals = [r[1] * 100 for r in rows] + [b[1] * 100 for b in brows]
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["darkorange"] * len(rows) + ["crimson"] * len(brows)
ax.barh(range(len(vals)), vals, color=colors)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels([l[:38] for l in labels], fontsize=8)
ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("portfolio impact (%)")
ax.set_title("Stress scenarios -- portfolio P&L (orange=historical, red=hypothetical)")
fig.tight_layout(); fig.savefig(os.path.join(C.OUT_DIR, "stress_test.png"), dpi=130)
plt.close(fig)
print("\nSaved: outputs/stress_test_results.csv, outputs/stress_test.png")
