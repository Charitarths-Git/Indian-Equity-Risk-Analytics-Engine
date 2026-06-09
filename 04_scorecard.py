"""
STAGE 4 -- Model scorecard: the PROPER metrics for judging a VaR/CVaR model.

Why not F1/recall/precision?
----------------------------
VaR is a QUANTILE forecast, not a class label. The goal is CALIBRATION
(breach rate == alpha), not catching every loss. A model that always predicts
a giant VaR has recall=1 but is worthless. So we score with:

  CALIBRATION  (is the breach rate right?)
    * exception (hit) rate vs expected alpha
    * Kupiec POF .............. unconditional coverage  (chi2_1)
    * Christoffersen IND ...... breaches independent / not clustered (chi2_1)
    * Christoffersen CC ....... joint coverage+independence (chi2_2)
    * Dynamic Quantile (DQ) ... Engle-Manganelli, most powerful (chi2)
    * Basel traffic-light zone

  SKILL  (which model forecasts the quantile best? -> rank by this)
    * average PINBALL / quantile loss  (the consistent scoring fn for VaR)
    * Lopez magnitude loss  (penalises breaches by size)

  TAIL SEVERITY (is CVaR right?)
    * ES ratio: mean realised tail loss / predicted CVaR on breach days (~1 good)

  VOLATILITY SUB-MODEL (here regression metrics DO apply)
    * QLIKE & RMSE of the variance forecast vs realised r^2  (GARCH vs constant)

Run:  python 04_scorecard.py
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

# ----------------------------- data ----------------------------------------
port = pd.read_csv(C.PORT_RET_CSV, index_col=0, parse_dates=True)["portfolio_return"]
train = port[port.index < C.TRAIN_END]
full = port
test_dates = full.index[(full.index >= C.TRAIN_END) & (full.index <= C.BACKTEST_END)]

# GARCH params from train; volatility filtered forward (no look-ahead)
res_tr = rm.fit_garch(train); gp = rm.garch_params(res_tr)
sigma_arr, _ = rm.garch_conditional_sigma(gp, full)
sigma_by_date = pd.Series(sigma_arr, index=full.index)

METHODS = ["Historical", "Normal", "Student-t", "GARCH-t"]


def predict(method, alpha, d):
    prior = full[full.index < d].values
    if method == "Historical":
        return rm.historical_var_cvar(prior, alpha)
    if method == "Normal":
        return rm.normal_var_cvar(prior, alpha)
    if method == "Student-t":
        v, c, _ = rm.t_var_cvar(prior, alpha)
        return v, c
    return rm.garch_var_cvar(gp["mu"], sigma_by_date[d], gp["nu"], alpha)


# ----------------------------- metric helpers ------------------------------
def pinball_loss(r, q, tau):
    """Consistent scoring fn for the tau-quantile q. Lower = better."""
    return np.mean((r - q) * (tau - (r < q).astype(float)))


def lopez_loss(r, q):
    """1 + squared breach magnitude on exceptions, else 0. Lower = better."""
    return np.mean(np.where(r < q, 1.0 + (q - r) ** 2, 0.0))


def dq_test(r, q, alpha, lags=4):
    """Engle-Manganelli Dynamic Quantile test p-value."""
    hit = (r < q).astype(float) - alpha
    N = len(hit)
    cols = [np.ones(N - lags)]
    for l in range(1, lags + 1):
        cols.append(hit[lags - l:N - l])
    cols.append((-q)[lags:])               # current VaR level
    X = np.column_stack(cols)
    Y = hit[lags:]
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        stat = float(beta @ (X.T @ X) @ beta) / (alpha * (1 - alpha))
        return chi2.sf(stat, X.shape[1])
    except Exception:
        return np.nan


def basel_zone(x, N, alpha):
    """Basel traffic light, scaled to the test length (expected = alpha*N)."""
    exp = alpha * N
    if x <= exp + 2 * np.sqrt(exp):   # rough green band
        return "GREEN"
    if x <= exp + 4 * np.sqrt(exp):
        return "YELLOW"
    return "RED"


# ----------------------------- run scorecard -------------------------------
rows = []
for cl in C.CONFIDENCE_LEVELS:
    alpha = 1 - cl
    for m in METHODS:
        V, CV, R = [], [], []
        for d in test_dates:
            v, c = predict(m, alpha, d)
            V.append(v); CV.append(c); R.append(full.loc[d])
        V, CV, R = np.array(V), np.array(CV), np.array(R)
        q = -V                                   # predicted return quantile
        hits = (R < q).astype(int)
        N, x = len(R), int(hits.sum())

        lr_uc, p_uc = rm.kupiec_pof(N, x, alpha)
        lr_ind, p_ind = rm.christoffersen_independence(hits)
        if not np.isnan(lr_ind):
            p_cc = chi2.sf(lr_uc + lr_ind, 2)
        else:
            p_cc = np.nan
        es_ratio = ((-R[hits == 1]).mean() / CV[hits == 1].mean()
                    if x > 0 else np.nan)

        rows.append({
            "conf": f"{int(cl*100)}%", "method": m,
            "hit_rate": round(100 * x / N, 2),         # %
            "expected": round(100 * alpha, 2),         # %
            "exceptions": x,
            "kupiec_p": round(p_uc, 3),
            "christ_ind_p": (round(p_ind, 3) if not np.isnan(p_ind) else None),
            "christ_cc_p": (round(p_cc, 3) if not np.isnan(p_cc) else None),
            "DQ_p": (round(dq_test(R, q, alpha), 3)),
            "pinball_x1e4": round(pinball_loss(R, q, alpha) * 1e4, 4),
            "lopez": round(lopez_loss(R, q), 4),
            "ES_ratio": (round(es_ratio, 2) if not np.isnan(es_ratio) else None),
            "basel": basel_zone(x, N, alpha),
        })

score = pd.DataFrame(rows)
score.to_csv(os.path.join(C.OUT_DIR, "model_scorecard.csv"), index=False)

pd.set_option("display.width", 200, "display.max_columns", 20)
print("=" * 100)
print("MODEL SCORECARD  --  out-of-sample", test_dates.min().date(), "->",
      test_dates.max().date(), f"({len(test_dates)} days)")
print("=" * 100)
print(score.to_string(index=False))

# ----- ranking by skill (pinball loss) -----
print("\nRANKING by quantile (pinball) loss  -- lower = better forecast:")
for cl in C.CONFIDENCE_LEVELS:
    sub = score[score.conf == f"{int(cl*100)}%"].sort_values("pinball_x1e4")
    order = " > ".join(f"{r.method}({r.pinball_x1e4})" for _, r in sub.iterrows())
    print(f"  {int(cl*100)}%:  {order}")

# ----------------------------- volatility sub-model ------------------------
print("\n" + "=" * 100)
print("VOLATILITY-FORECAST SCORE (GARCH vs constant-vol benchmark) -- "
      "regression metrics DO apply here")
print("=" * 100)
oos = full.loc[test_dates]
r2 = np.maximum(oos.values ** 2, 1e-10)          # noisy proxy for true variance
garch_var = sigma_by_date.loc[test_dates].values ** 2
const_var = np.full(len(oos), train.std() ** 2)


def qlike(proxy, f):
    ratio = proxy / f
    return np.mean(ratio - np.log(ratio) - 1)


def rmse(proxy, f):
    return np.sqrt(np.mean((proxy - f) ** 2))


print(f"  {'model':<14}{'QLIKE':>12}{'RMSE(var)':>14}")
print(f"  {'GARCH(1,1)-t':<14}{qlike(r2, garch_var):>12.4f}"
      f"{rmse(r2, garch_var)*1e6:>14.3f}")
print(f"  {'Constant vol':<14}{qlike(r2, const_var):>12.4f}"
      f"{rmse(r2, const_var)*1e6:>14.3f}")
print("  (lower QLIKE = better variance forecast; RMSE in 1e-6 var units)")
print("\nSaved: outputs/model_scorecard.csv")
