"""
STAGE 6 -- Filtered Historical Simulation (FHS): the upgrade.

FHS = GARCH conditional volatility  x  bootstrapped EMPIRICAL standardised
residuals. It keeps the time-varying vol but drops the Normal/t tail assumption,
using the real (fat, skewed) residual shape instead.

This script:
  (a) Forecasts TODAY's FHS VaR/CVaR (1-day exact + 1-week sqrt-time + 1-week
      proper path-simulation).
  (b) Backtests FHS vs GARCH-t vs Normal out-of-sample, focusing on whether FHS
      fixes the 95% under-coverage seen earlier.
  (c) Re-runs the predicted-vs-actual coverage check for FHS.

Run:  python 06_fhs.py
"""

import os
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import chi2

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
    return f"Rs {'-' if neg else ''}{s}"


port = pd.read_csv(C.PORT_RET_CSV, index_col=0, parse_dates=True)["portfolio_return"]
train = port[port.index < C.TRAIN_END]
full = port

# ---- GARCH on TRAIN (for backtest, no look-ahead) ----
res_tr = rm.fit_garch(train); gp_tr = rm.garch_params(res_tr)
sig_tr_full, _ = rm.garch_conditional_sigma(gp_tr, full)
sig_tr_full = pd.Series(sig_tr_full, index=full.index)
z_tr_full = pd.Series(rm.standardized_residuals(gp_tr["mu"], sig_tr_full.values, full),
                      index=full.index)

# ---- GARCH on FULL (for today's live forecast) ----
res_full = rm.fit_garch(full); gp_full = rm.garch_params(res_full)
sig_full_arr, sig_next = rm.garch_conditional_sigma(gp_full, full)
z_full = rm.standardized_residuals(gp_full["mu"], sig_full_arr, full)

print("=" * 78)
print(f"FILTERED HISTORICAL SIMULATION  --  today's forecast ({C.TODAY})")
print("=" * 78)
print(f"GARCH next-day vol forecast: {sig_next*100:.2f}%  | residual pool n={len(z_full)}, "
      f"resid skew={pd.Series(z_full).skew():+.2f}, resid kurt={pd.Series(z_full).kurt():+.2f}")

for cl in C.CONFIDENCE_LEVELS:
    a = 1 - cl
    v1, c1 = rm.fhs_var_cvar(gp_full["mu"], sig_next, z_full, a)          # 1-day exact
    vw_sqrt = rm.scale_to_horizon(v1, 5); cw_sqrt = rm.scale_to_horizon(c1, 5)
    vw_path, cw_path = rm.fhs_path_var_cvar(gp_full, sig_next, z_full, a, 5,
                                            C.MC_SIMULATIONS, C.MC_SEED)
    print(f"\n  {int(cl*100)}% confidence")
    print(f"    1-day        VaR {inr(v1*PV):>14}   CVaR {inr(c1*PV):>14}")
    print(f"    1-week (sqrt)VaR {inr(vw_sqrt*PV):>14}   CVaR {inr(cw_sqrt*PV):>14}")
    print(f"    1-week (path)VaR {inr(vw_path*PV):>14}   CVaR {inr(cw_path*PV):>14}"
          f"   <- proper multi-day FHS")


# ---------------- backtest: FHS vs GARCH-t vs Normal ----------------------
print("\n" + "=" * 78)
print("BACKTEST  --  does FHS fix the 95% under-coverage?  (OOS 2 Mar -> 6 Jun)")
print("=" * 78)

test_dates = full.index[(full.index >= C.TRAIN_END) & (full.index <= C.BACKTEST_END)]


def pinball(r, q, tau):
    return np.mean((r - q) * (tau - (r < q).astype(float)))


rows = []
for cl in C.CONFIDENCE_LEVELS:
    a = 1 - cl
    for method in ["Normal", "GARCH-t", "FHS"]:
        V, R = [], []
        for d in test_dates:
            prior = full[full.index < d]
            if method == "Normal":
                v, _ = rm.normal_var_cvar(prior.values, a)
            elif method == "GARCH-t":
                v, _ = rm.garch_var_cvar(gp_tr["mu"], sig_tr_full[d], gp_tr["nu"], a)
            else:  # FHS: empirical residual pool from days < d, scaled by sigma_d
                pool = z_tr_full[z_tr_full.index < d].values
                v, _ = rm.fhs_var_cvar(gp_tr["mu"], sig_tr_full[d], pool, a)
            V.append(v); R.append(full.loc[d])
        V, R = np.array(V), np.array(R)
        q = -V
        hits = (R < q).astype(int)
        N, x = len(R), int(hits.sum())
        _, p_uc = rm.kupiec_pof(N, x, a)
        rows.append({
            "conf": f"{int(cl*100)}%", "method": method,
            "hit_rate_%": round(100 * x / N, 2), "target_%": round(100 * a, 2),
            "exceptions": x, "kupiec_p": round(p_uc, 3),
            "pinball_x1e4": round(pinball(R, q, a) * 1e4, 3),
            "verdict": "PASS" if p_uc > 0.05 else "FAIL",
        })

bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(C.OUT_DIR, "fhs_backtest_comparison.csv"), index=False)
print("\n", bt.to_string(index=False))


# ---------------- predicted vs actual coverage for FHS (to today) ---------
print("\n" + "=" * 78)
print("PREDICTED vs ACTUAL coverage  --  FHS, 2 Mar -> 2026-06-08")
print("=" * 78)
oos = full.index[full.index >= C.TRAIN_END]
for cl in C.CONFIDENCE_LEVELS:
    a = 1 - cl
    within = 0
    for d in oos:
        pool = z_tr_full[z_tr_full.index < d].values
        v, _ = rm.fhs_var_cvar(gp_tr["mu"], sig_tr_full[d], pool, a)
        if full.loc[d] >= -v:
            within += 1
    n = len(oos)
    print(f"  {int(cl*100)}% VaR : actual within {within}/{n} = {100*within/n:.1f}%  "
          f"(target {cl*100:.1f}%)")

print("\nSaved: outputs/fhs_backtest_comparison.csv")
