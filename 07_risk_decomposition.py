"""
TIER-1 (1) -- Component / Marginal VaR  (risk decomposition, Euler allocation).

Total VaR is split into each stock's CONTRIBUTION, which answers the decision
question: "which position is carrying the risk -- what do I trim/hedge?"

Theory (Gaussian, zero-mean approx; VaR = z * sigma_p):
  sigma_p          = sqrt(w' S w)
  Marginal VaR_i   = z * (S w)_i / sigma_p           (d VaR / d w_i)
  Component VaR_i  = w_i * Marginal VaR_i             (SUMS to total VaR -- Euler)
  % contribution_i = Component VaR_i / VaR = w_i (S w)_i / (w' S w)

A stock whose risk% >> weight% is a CONCENTRATOR; risk% < weight% is a diversifier.

Run:  python 07_risk_decomposition.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import config as C
import utils

PV = C.PORTFOLIO_VALUE_INR
port, assets, wdf = utils.load()
tickers = list(assets.columns)
w = wdf.loc[tickers, "weight"].values
sector = wdf.loc[tickers, "sector"].values
S = assets.cov().values                      # daily covariance

cl = 0.99
z = -norm.ppf(1 - cl)                         # positive 2.326 for 99%
sig_p = np.sqrt(w @ S @ w)
Sw = S @ w
marg = z * Sw / sig_p                         # marginal VaR (per unit weight)
comp = w * marg                               # component VaR (decimal); sums to total
total_var = z * sig_p                         # = comp.sum()
pct = comp / total_var                        # % risk contribution

standalone = z * w * np.sqrt(np.diag(S))      # each position's VaR ignoring correlation

df = pd.DataFrame({
    "sector": sector,
    "weight_%": (w * 100).round(2),
    "risk_contrib_%": (pct * 100).round(2),
    "risk/weight": (pct / w).round(2),
    "component_VaR_inr": (comp * PV).round(0),
    "standalone_VaR_inr": (standalone * PV).round(0),
}, index=tickers).sort_values("risk_contrib_%", ascending=False)
df.index.name = "ticker"
df.to_csv(os.path.join(C.OUT_DIR, "risk_decomposition.csv"))

diversification_benefit = 1 - total_var / (standalone.sum())

print("=" * 78)
print(f"RISK DECOMPOSITION  --  1-day 99% VaR = {utils.inr(total_var*PV)}  "
      f"(portfolio {utils.inr(PV)})")
print("=" * 78)
print(df.to_string())
print(f"\nSum of component VaRs = {utils.inr(comp.sum()*PV)}  (== total VaR, Euler check)")
print(f"Sum of STANDALONE VaRs = {utils.inr(standalone.sum()*PV)}  (if held uncorrelated)")
print(f"Diversification benefit = {diversification_benefit*100:.1f}%  "
      f"(correlation saves this much risk)")

top = df.index[0]
conc = df[df["risk/weight"] > 1.0]
print(f"\nBIGGEST RISK DRIVER : {top.replace('.NS','')} -- "
      f"{df.loc[top,'risk_contrib_%']}% of risk on {df.loc[top,'weight_%']}% of capital.")
print("CONCENTRATORS (risk% > weight%, trim/hedge candidates): "
      + ", ".join(t.replace('.NS','') for t in conc.index))

# chart: weight vs risk contribution
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(df))
ax.bar(x - 0.2, df["weight_%"], width=0.4, label="weight %", color="steelblue")
ax.bar(x + 0.2, df["risk_contrib_%"], width=0.4, label="risk contribution %",
       color="crimson", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([t.replace(".NS", "") for t in df.index],
                                     rotation=90, fontsize=8)
ax.set_ylabel("%"); ax.set_title("Capital weight vs risk contribution "
                                 "(bars where red > blue = risk concentrators)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(C.OUT_DIR, "risk_decomposition.png"), dpi=130)
plt.close(fig)
print("\nSaved: outputs/risk_decomposition.csv, outputs/risk_decomposition.png")
