"""Shared statistical utilities for all validation phases.

Implements the pre-specified SAP (Statistical Analysis Plan):
    - Wilson confidence intervals for proportions
    - Paired bootstrap of median differences
    - McNemar test for paired proportions
    - ICC computation
    - Bland–Altman statistics
    - Mixed-effects helpers
    - AUC under degradation curves
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "median",
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a univariate statistic.

    Parameters
    ----------
    data : array-like, 1-D
        Observed values.
    statistic : {"median", "mean"}
        Statistic to bootstrap.
    n_bootstrap : int
        Number of resamples.
    ci_level : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    point_est, ci_lower, ci_upper : float
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan

    stat_fn = np.median if statistic == "median" else np.mean
    point_est = float(stat_fn(data))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point_est, ci_lower, ci_upper


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    statistic: str = "median",
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    r"""Bootstrap CI for \Delta = stat(a) - stat(b) on paired observations.

    Parameters
    ----------
    a, b : array-like, 1-D
        Paired observations (same length).

    Returns
    -------
    delta, ci_lower, ci_upper : float
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b), "Paired arrays must have equal length"
    n = len(a)

    stat_fn = np.median if statistic == "median" else np.mean
    delta = float(stat_fn(a) - stat_fn(b))

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = stat_fn(a[idx]) - stat_fn(b[idx])

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    return delta, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Wilson score CI for proportions (Phase A)
# ---------------------------------------------------------------------------


def wilson_ci(
    k: int,
    n: int,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    ci_level : float
        Confidence level.

    Returns
    -------
    p_hat, ci_lower, ci_upper : float
    """
    from scipy.stats import norm

    if n == 0:
        return np.nan, np.nan, np.nan

    z = norm.ppf(1 - (1 - ci_level) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return float(p_hat), float(np.clip(lower, 0, 1)), float(np.clip(upper, 0, 1))


# ---------------------------------------------------------------------------
# McNemar test (Phase A — paired pipeline comparison)
# ---------------------------------------------------------------------------


def mcnemar_test(
    a_success: np.ndarray,
    b_success: np.ndarray,
) -> Tuple[float, float]:
    """McNemar test on paired binary success vectors.

    Returns
    -------
    chi2, p_value : float
    """
    from scipy.stats import chi2 as chi2_dist

    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    # b succeeds where a fails, and vice-versa
    b_not_a = int(np.sum(b & ~a))
    a_not_b = int(np.sum(a & ~b))

    if (b_not_a + a_not_b) == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b_not_a - a_not_b) - 1) ** 2 / (b_not_a + a_not_b)
    p_value = 1.0 - chi2_dist.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)


# ---------------------------------------------------------------------------
# ICC (Phase D)
# ---------------------------------------------------------------------------


def icc_31(
    measurements: np.ndarray,
) -> Tuple[float, float, float]:
    """ICC(3,1) — two-way mixed, single measures, consistency.

    Parameters
    ----------
    measurements : ndarray, shape (n_subjects, n_methods)
        Each row is a subject; each column is a measurement method / rater.

    Returns
    -------
    icc, ci_lower, ci_upper : float
        ICC estimate and approximate 95% CI.
    """
    from scipy.stats import f as f_dist

    measurements = np.asarray(measurements, dtype=float)
    n, k = measurements.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan

    grand_mean = measurements.mean()
    ssw = np.sum((measurements - measurements.mean(axis=1, keepdims=True)) ** 2)
    ssb = k * np.sum((measurements.mean(axis=1) - grand_mean) ** 2)
    msr = ssb / (n - 1)
    mse = ssw / (n * (k - 1) - (n - 1))  # simplified for ICC(3,1)

    # Avoid division by zero
    if (msr + (k - 1) * mse) == 0:
        return 0.0, 0.0, 0.0

    icc = (msr - mse) / (msr + (k - 1) * mse)

    # Approximate F-test based CI
    f_value = msr / max(mse, 1e-12)
    df1 = n - 1
    df2 = df1 * (k - 1)
    f_lower = f_value / f_dist.ppf(0.975, df1, df2)
    f_upper = f_value / f_dist.ppf(0.025, df1, df2)
    ci_lower = (f_lower - 1) / (f_lower + k - 1)
    ci_upper = (f_upper - 1) / (f_upper + k - 1)

    return float(np.clip(icc, -1, 1)), float(np.clip(ci_lower, -1, 1)), float(np.clip(ci_upper, -1, 1))


# ---------------------------------------------------------------------------
# Bland–Altman (Phase D)
# ---------------------------------------------------------------------------


def bland_altman(
    method_a: np.ndarray,
    method_b: np.ndarray,
) -> Dict[str, float]:
    """Bland–Altman agreement statistics.

    Parameters
    ----------
    method_a, method_b : array-like
        Paired measurements.

    Returns
    -------
    dict with keys: mean_diff, sd_diff, loa_lower, loa_upper, mean_of_means
    """
    a, b = np.asarray(method_a, dtype=float), np.asarray(method_b, dtype=float)
    diff = a - b
    mean_of_means = (a + b) / 2.0

    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return {
        "mean_diff": md,
        "sd_diff": sd,
        "loa_lower": md - 1.96 * sd,
        "loa_upper": md + 1.96 * sd,
        "mean_of_means": float(np.mean(mean_of_means)),
    }


# ---------------------------------------------------------------------------
# AUC under degradation curve (Phase C)
# ---------------------------------------------------------------------------


def degradation_auc(
    severity_levels: Sequence[float],
    metric_values: Sequence[float],
) -> float:
    """Trapezoid-rule AUC of metric vs perturbation severity.

    Higher AUC → more robust (assuming metric is "higher is better").

    Parameters
    ----------
    severity_levels : sequence of float
        Monotonically increasing perturbation severity.
    metric_values : sequence of float
        Corresponding performance metric at each severity level.

    Returns
    -------
    float
        AUC value.
    """
    x = np.asarray(severity_levels, dtype=float)
    y = np.asarray(metric_values, dtype=float)
    # Normalize x to [0,1]
    x_range = x[-1] - x[0]
    if x_range == 0:
        return float(y[0])
    x_norm = (x - x[0]) / x_range
    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", np.trapz)
    return float(_trapz(y, x_norm))


# ---------------------------------------------------------------------------
# Sample size helpers
# ---------------------------------------------------------------------------


def sample_size_ci_proportion(
    target_width: float = 0.10,
    p_est: float = 0.95,
    ci_level: float = 0.95,
) -> int:
    """Approximate sample size for a CI of given width around a proportion.

    Uses normal approximation: n ≈ z² p(1-p) / (w/2)².

    Parameters
    ----------
    target_width : float
        Total CI width (e.g. 0.10 for ±5%).
    p_est : float
        Estimated proportion.
    ci_level : float
        Confidence level.

    Returns
    -------
    int
        Required sample size (rounded up).
    """
    from scipy.stats import norm
    import math

    z = norm.ppf(1 - (1 - ci_level) / 2)
    n = z ** 2 * p_est * (1 - p_est) / (target_width / 2) ** 2
    return int(math.ceil(n))


def sample_size_paired_continuous(
    delta: float = 0.02,
    sigma_delta: float = 0.06,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate sample size for paired continuous comparison.

    n ≈ ((z_α/2 + z_β) σ_Δ / δ)²

    Parameters
    ----------
    delta : float
        Minimum detectable effect.
    sigma_delta : float
        Standard deviation of paired differences.
    alpha, power : float
        Significance level and desired power.

    Returns
    -------
    int
        Required sample size (rounded up).
    """
    from scipy.stats import norm
    import math

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) * sigma_delta / delta) ** 2
    return int(math.ceil(n))
