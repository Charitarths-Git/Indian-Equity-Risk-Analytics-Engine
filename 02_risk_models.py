"""
STAGE 2 -- Risk-model library (pure functions, no I/O).

Conventions
-----------
* `returns` are SIMPLE daily returns as decimals (e.g. -0.012 = -1.2%).
* VaR and CVaR are returned as POSITIVE loss fractions of portfolio value.
      VaR_frac = 0.025  ->  a 2.5% loss.
* alpha = 1 - confidence  (tail probability; e.g. 0.01 for 99%).
* 1-week (h-day) figures use the square-root-of-time rule:  VaR_h = VaR_1 * sqrt(h).
      (Valid under i.i.d. returns; we flag the caveat in the report.)
"""

import numpy as np
from scipy import stats

SQRT = np.sqrt


# ---------------------------------------------------------------------------
# generic tail helper -- mean of the distribution's quantiles in (0, alpha].
# Works for ANY distribution given its inverse-CDF (ppf). Used for CVaR.
# ---------------------------------------------------------------------------
def _tail_quantile_mean(ppf, alpha, n=4000):
    us = np.linspace(1e-6, alpha, n)
    return ppf(us).mean()          # E[X | X <= q_alpha]  (a negative number)


# ---------------------------------------------------------------------------
# 1. HISTORICAL SIMULATION  (non-parametric)
# ---------------------------------------------------------------------------
def historical_var_cvar(returns, alpha):
    q = np.quantile(returns, alpha)            # alpha-quantile of returns (<0)
    var = -q
    tail = returns[returns <= q]
    cvar = -tail.mean() if len(tail) else var
    return var, cvar


# ---------------------------------------------------------------------------
# 2. PARAMETRIC -- Gaussian (variance-covariance), closed form
# ---------------------------------------------------------------------------
def normal_var_cvar(returns, alpha):
    mu = returns.mean()
    sd = returns.std(ddof=1)
    z = stats.norm.ppf(alpha)                  # negative
    var = -(mu + sd * z)
    # closed-form Gaussian Expected Shortfall
    cvar = -(mu - sd * stats.norm.pdf(z) / alpha)
    return var, cvar


# ---------------------------------------------------------------------------
# 3. PARAMETRIC -- Student-t (fat tails), MLE-fitted
# ---------------------------------------------------------------------------
def t_var_cvar(returns, alpha):
    nu, loc, scale = stats.t.fit(returns)      # MLE fit
    var = -stats.t.ppf(alpha, nu, loc, scale)
    cvar = -_tail_quantile_mean(
        lambda u: stats.t.ppf(u, nu, loc, scale), alpha)
    return var, cvar, nu


# ---------------------------------------------------------------------------
# 4. MONTE CARLO -- asset-level multivariate Normal (uses the covariance matrix)
#    Simulate correlated asset returns via Cholesky, aggregate with weights,
#    then read the empirical tail of the simulated portfolio P&L.
# ---------------------------------------------------------------------------
def mc_var_cvar(asset_returns, weights, alpha, n_sims, seed):
    mu = asset_returns.mean().values                 # (k,)
    cov = asset_returns.cov().values                 # (k,k)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mu)))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(mu)))
    sim_assets = mu + z @ L.T                         # (n_sims, k) correlated
    sim_port = sim_assets @ weights                   # (n_sims,)
    q = np.quantile(sim_port, alpha)
    var = -q
    cvar = -sim_port[sim_port <= q].mean()
    return var, cvar


# ---------------------------------------------------------------------------
# 5. GARCH(1,1)-t  -- conditional (time-varying) volatility
# ---------------------------------------------------------------------------
def fit_garch(returns_decimal):
    """Fit GARCH(1,1) with Student-t innovations. Returns the arch result.
    arch prefers returns in PERCENT for numerical stability."""
    from arch import arch_model
    y = returns_decimal * 100.0
    am = arch_model(y, mean="Constant", vol="Garch", p=1, q=1, dist="t")
    return am.fit(disp="off")


def garch_params(res):
    p = res.params
    return dict(
        mu=p["mu"] / 100.0,                       # back to decimal
        omega=p["omega"], a1=p["alpha[1]"], b1=p["beta[1]"],
        nu=p["nu"],
    )


def garch_conditional_sigma(params, returns_decimal):
    """1-step-ahead conditional volatility (decimal) for every day, plus the
    next-day forecast. sigma_t uses information through t-1 only.

    Returns: (sigma_series_decimal[len T], sigma_next_decimal)
    """
    y = returns_decimal.values * 100.0            # percent
    mu_pct = params["mu"] * 100.0
    w, a, b = params["omega"], params["a1"], params["b1"]
    eps = y - mu_pct
    T = len(y)
    s2 = np.empty(T)
    s2[0] = w / max(1e-8, (1 - a - b))            # unconditional variance seed
    for i in range(1, T):
        s2[i] = w + a * eps[i - 1] ** 2 + b * s2[i - 1]
    sigma_t = np.sqrt(s2) / 100.0                 # decimal, 1-step-ahead per day
    # forecast for the day AFTER the last observation
    s2_next = w + a * eps[-1] ** 2 + b * s2[-1]
    return sigma_t, np.sqrt(s2_next) / 100.0


def _std_t_quantile(alpha, nu):
    """alpha-quantile of a Student-t standardised to UNIT variance."""
    return stats.t.ppf(alpha, nu) * SQRT((nu - 2) / nu)


def _std_t_tail_mean(alpha, nu, n=4000):
    """E[Z | Z <= q_alpha] for unit-variance Student-t (negative)."""
    us = np.linspace(1e-6, alpha, n)
    zs = stats.t.ppf(us, nu) * SQRT((nu - 2) / nu)
    return zs.mean()


def garch_var_cvar(mu, sigma, nu, alpha):
    """Conditional VaR/CVaR given forecast sigma (decimal) and t-dof nu."""
    zq = _std_t_quantile(alpha, nu)
    zes = _std_t_tail_mean(alpha, nu)
    var = -(mu + sigma * zq)
    cvar = -(mu + sigma * zes)
    return var, cvar


# ---------------------------------------------------------------------------
# 6. FILTERED HISTORICAL SIMULATION (FHS)
#    GARCH conditional vol  x  EMPIRICAL (bootstrapped) standardised residuals.
#    Combines time-varying vol with the real fat-tailed/skewed residual shape,
#    instead of assuming Normal or t innovations.
# ---------------------------------------------------------------------------
def standardized_residuals(mu, sigma_series_decimal, returns_decimal):
    """z_t = (r_t - mu) / sigma_t  -- the i.i.d.-ish shocks GARCH filters out."""
    return (returns_decimal.values - mu) / sigma_series_decimal


def fhs_var_cvar(mu, sigma_next, z_pool, alpha):
    """1-day FHS: scale the empirical residual quantile by today's vol forecast.
    Exact (no MC noise): the simulated returns are {mu + sigma_next * z}."""
    za = np.quantile(z_pool, alpha)
    var = -(mu + sigma_next * za)
    tail = z_pool[z_pool <= za]
    es = tail.mean() if len(tail) else za
    cvar = -(mu + sigma_next * es)
    return var, cvar


def fhs_path_var_cvar(gp, sigma_next_dec, z_pool, alpha, h, n_sims, seed):
    """Proper multi-day FHS via path simulation: evolve the GARCH variance
    forward with bootstrapped residuals and compound the h-day return.
    `gp` is the dict from garch_params(); units handled in percent internally."""
    rng = np.random.default_rng(seed)
    mu_p = gp["mu"] * 100.0
    w, a, b = gp["omega"], gp["a1"], gp["b1"]
    s2 = np.full(n_sims, (sigma_next_dec * 100.0) ** 2)   # next-day variance (pct^2)
    cum = np.zeros(n_sims)
    for _ in range(h):
        z = rng.choice(z_pool, size=n_sims, replace=True)  # empirical shocks
        eps = np.sqrt(s2) * z
        cum += (mu_p + eps)                                # accumulate pct return
        s2 = w + a * eps ** 2 + b * s2                     # GARCH update
    sim = cum / 100.0                                      # back to decimal
    q = np.quantile(sim, alpha)
    var = -q
    cvar = -sim[sim <= q].mean()
    return var, cvar


# ---------------------------------------------------------------------------
# horizon scaling (square-root-of-time)
# ---------------------------------------------------------------------------
def scale_to_horizon(value_1d, h):
    return value_1d * SQRT(h)


# ---------------------------------------------------------------------------
# BACKTEST STATISTICS
# ---------------------------------------------------------------------------
def kupiec_pof(n, x, p):
    """Unconditional-coverage LR test. n=obs, x=exceptions, p=expected rate.
    Returns (LR, p_value)."""
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
        return lr, stats.chi2.sf(lr, 1)
    pi = x / n
    lr = -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
        - (n - x) * np.log(1 - pi) - x * np.log(pi)
    )
    return lr, stats.chi2.sf(lr, 1)


def christoffersen_independence(hits):
    """Independence of exceptions (no clustering). hits = 0/1 array."""
    hits = np.asarray(hits).astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits[:-1], hits[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1
    # transition probabilities
    denom0, denom1 = (n00 + n01), (n10 + n11)
    if denom0 == 0 or denom1 == 0:
        return np.nan, np.nan
    pi01 = n01 / denom0
    pi11 = n11 / denom1
    pi = (n01 + n11) / (denom0 + denom1)
    if pi in (0, 1) or pi01 in (0,) or pi11 in (0,):
        # degenerate (too few exceptions) -> test not informative
        return np.nan, np.nan
    num = ((1 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
    den = ((1 - pi01) ** n00) * (pi01 ** n01) * ((1 - pi11) ** n10) * (pi11 ** n11)
    lr = -2 * np.log(num / den)
    return lr, stats.chi2.sf(lr, 1)
