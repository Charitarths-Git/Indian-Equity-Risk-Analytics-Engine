"""
STAGE 5 -- Predicted VaR vs ACTUAL realised P&L, day by day.

A VaR model is NOT a point forecast of the return -- it forecasts a LOSS
BOUNDARY. "Match" therefore means: did the actual daily loss stay INSIDE the
predicted VaR, and did it breach only ~(1-confidence) of the time?

For every day from 2 Mar 2026 to the latest actual (2026-06-08) we show the
1-day VaR predicted using ONLY prior information vs what actually happened.

Run:  python 05_actual_vs_predicted.py
"""

import os
import importlib.util
import numpy as np
import pandas as pd

import config as C
_spec = importlib.util.spec_from_file_location(
    "risk_models", os.path.join(C.ROOT, "02_risk_models.py"))
rm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rm)

PV = C.PORTFOLIO_VALUE_INR


def inr(x):
    x = int(round(x)); neg = x < 0; s = str(abs(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]; parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        parts.insert(0, head); s = ",".join(parts) + "," + tail
    return f"{'-' if neg else ''}{s}"


port = pd.read_csv(C.PORT_RET_CSV, index_col=0, parse_dates=True)["portfolio_return"]
train = port[port.index < C.TRAIN_END]
full = port

# GARCH params from train; volatility filtered forward (no look-ahead)
res_tr = rm.fit_garch(train); gp = rm.garch_params(res_tr)
sigma_arr, _ = rm.garch_conditional_sigma(gp, full)
sigma_by_date = pd.Series(sigma_arr, index=full.index)

# every actual trading day from the train cut to the latest observation (incl today)
oos = full.index[(full.index >= C.TRAIN_END)]

rows = []
for d in oos:
    a99, a95 = 0.01, 0.05
    v99, c99 = rm.garch_var_cvar(gp["mu"], sigma_by_date[d], gp["nu"], a99)
    v95, _ = rm.garch_var_cvar(gp["mu"], sigma_by_date[d], gp["nu"], a95)
    r = full.loc[d]
    rows.append({
        "date": d.date(),
        "pred_VaR99_inr": round(v99 * PV),
        "pred_VaR95_inr": round(v95 * PV),
        "pred_CVaR99_inr": round(c99 * PV),
        "actual_return_%": round(r * 100, 2),
        "actual_PnL_inr": round(r * PV),
        "within_99": "yes" if r >= -v99 else "BREACH",
        "within_95": "yes" if r >= -v95 else "BREACH",
    })

cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(C.OUT_DIR, "actual_vs_predicted.csv"), index=False)

n = len(cmp)
b99 = (cmp.within_99 == "BREACH").sum()
b95 = (cmp.within_95 == "BREACH").sum()

print("=" * 84)
print(f"PREDICTED VaR  vs  ACTUAL P&L   ({cmp.date.iloc[0]} -> {cmp.date.iloc[-1]}, "
      f"{n} trading days)")
print("=" * 84)
print("(VaR = predicted LOSS boundary in Rs; actual within it = model 'matched')\n")

print(f"{'date':<12}{'pred VaR99':>13}{'pred VaR95':>13}"
      f"{'actual P&L':>14}{'ret%':>8}{'  99%':>7}{'  95%':>7}")
for _, r in cmp.tail(14).iterrows():
    print(f"{str(r['date']):<12}{inr(r.pred_VaR99_inr):>13}{inr(r.pred_VaR95_inr):>13}"
          f"{inr(r.actual_PnL_inr):>14}{r['actual_return_%']:>8}"
          f"{r['within_99']:>7}{r['within_95']:>7}")

print("\n--- COVERAGE (does actual stay inside predicted VaR at the right rate?) ---")
print(f"  99% VaR : actual stayed within {n-b99}/{n} days = {100*(n-b99)/n:.1f}%  "
      f"(target 99.0%) ; breaches={b99}, expected ~{0.01*n:.1f}")
print(f"  95% VaR : actual stayed within {n-b95}/{n} days = {100*(n-b95)/n:.1f}%  "
      f"(target 95.0%) ; breaches={b95}, expected ~{0.05*n:.1f}")

# today's row + tomorrow's (unverifiable yet) forecast
today = cmp.iloc[-1]
print("\n--- TODAY (latest actual we have) ---")
print(f"  {today['date']}:  predicted 99% VaR = Rs {inr(today.pred_VaR99_inr)} loss boundary"
      f"  |  ACTUAL = {today['actual_return_%']:+.2f}%  (Rs {inr(today.actual_PnL_inr)})"
      f"  ->  {'WITHIN bound (matched)' if today.within_99=='yes' else 'BREACH'}")

_, sigma_next = rm.garch_conditional_sigma(gp, full)
vt, ct = rm.garch_var_cvar(gp["mu"], sigma_next, gp["nu"], 0.01)
print("\n--- TOMORROW (forward forecast, NOT yet verifiable) ---")
print(f"  next trading day: 99% VaR = Rs {inr(vt*PV)},  CVaR = Rs {inr(ct*PV)}")
print("\nSaved: outputs/actual_vs_predicted.csv  (full day-by-day table)")
