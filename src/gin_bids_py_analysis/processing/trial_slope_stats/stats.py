from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def compute_linear_regression_maps(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_features: int,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute per-feature, per-time OLS maps for y ~ x.

    Parameters
    ----------
    predictor_values:
        Array of shape ``(n_trials,)``.
    epochs:
        Array of shape ``(n_trials, n_features, n_times)``.

    Returns
    -------
    slope, intercept, r_value, p_value, stats_valid
        Arrays are ``(n_features, n_times)``; all NaN when invalid.
    """
    empty = np.full((n_features, n_times), np.nan, dtype=np.float64)

    x = np.asarray(predictor_values, dtype=np.float64).reshape(-1)
    y = np.asarray(epochs, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError("epochs must be 3-D (n_trials, n_features, n_times).")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"predictor/epochs mismatch: {x.shape[0]} predictor values for {y.shape[0]} epochs."
        )

    finite_x = np.isfinite(x)
    if not np.all(finite_x):
        x = x[finite_x]
        y = y[finite_x]

    n_trials = int(x.shape[0])
    if n_trials < 3:
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), False

    var_x = float(np.nanvar(x, ddof=1))
    if not np.isfinite(var_x) or var_x <= 0.0:
        return empty.copy(), empty.copy(), empty.copy(), empty.copy(), False

    # Flatten feature x time so regression is computed in one vectorized pass.
    y_flat = y.reshape(n_trials, -1)
    x_mean = float(np.nanmean(x))
    y_mean = np.nanmean(y_flat, axis=0, dtype=np.float64)

    x_centered = x - x_mean
    y_centered = y_flat - y_mean[np.newaxis, :]
    cov = np.sum(x_centered[:, np.newaxis] * y_centered, axis=0, dtype=np.float64) / float(n_trials - 1)

    slope_flat = cov / var_x
    intercept_flat = y_mean - (slope_flat * x_mean)

    var_y = np.nanvar(y_flat, axis=0, ddof=1, dtype=np.float64)
    denom = np.sqrt(var_x * var_y)

    r_flat = np.full_like(slope_flat, np.nan, dtype=np.float64)
    corr_mask = np.isfinite(denom) & (denom > 0.0)
    r_flat[corr_mask] = cov[corr_mask] / denom[corr_mask]
    r_flat[corr_mask] = np.clip(r_flat[corr_mask], -1.0, 1.0)

    p_flat = np.full_like(slope_flat, np.nan, dtype=np.float64)
    df = float(n_trials - 2)
    valid_t_mask = np.isfinite(r_flat) & (np.abs(r_flat) < 1.0)
    if np.any(valid_t_mask):
        t_values = r_flat[valid_t_mask] * np.sqrt(df / (1.0 - np.square(r_flat[valid_t_mask], dtype=np.float64)))
        p_flat[valid_t_mask] = 2.0 * student_t.sf(np.abs(t_values), df)

    perfect_mask = np.isfinite(r_flat) & (np.abs(r_flat) >= 1.0)
    p_flat[perfect_mask] = 0.0

    return (
        slope_flat.reshape(n_features, n_times),
        intercept_flat.reshape(n_features, n_times),
        r_flat.reshape(n_features, n_times),
        p_flat.reshape(n_features, n_times),
        True,
    )


def zscore_predictor_values(
    predictor_values: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Z-score a 1-D predictor across trials.

    Returns
    -------
    z_values, valid
        ``valid`` is False when fewer than 3 finite trials are available or when
        the predictor variance is not strictly positive.
    """
    x = np.asarray(predictor_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    x = x[finite]
    if x.shape[0] < 3:
        return np.full_like(np.asarray(predictor_values, dtype=np.float64).reshape(-1), np.nan), False
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return np.full_like(np.asarray(predictor_values, dtype=np.float64).reshape(-1), np.nan), False
    out = np.full_like(np.asarray(predictor_values, dtype=np.float64).reshape(-1), np.nan)
    out[finite] = (x - mean) / std
    return out, True


def zscore_predictor_values_by_scope(
    condition_a_values: np.ndarray,
    condition_b_values: np.ndarray,
    *,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Return predictor arrays z-scored with the requested scope.

    Parameters
    ----------
    condition_a_values, condition_b_values:
        Predictor vectors for the two conditions.
    scope:
        One of ``"none"``, ``"condition"``, or ``"global"``.
    """
    arr_a = np.asarray(condition_a_values, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(condition_b_values, dtype=np.float64).reshape(-1)
    mode = str(scope).strip().lower()

    if mode == "none":
        return arr_a.copy(), arr_b.copy(), True, True

    if mode == "condition":
        scaled_a, valid_a = _zscore_predictor_or_empty(arr_a)
        scaled_b, valid_b = _zscore_predictor_or_empty(arr_b)
        return scaled_a, scaled_b, valid_a, valid_b

    if mode != "global":
        raise ValueError(
            f"Unsupported predictor z-score scope: {scope!r}. "
            "Valid values are 'none', 'condition', and 'global'."
        )

    combined_parts = [arr for arr in (arr_a, arr_b) if arr.size > 0]
    if not combined_parts:
        return arr_a.copy(), arr_b.copy(), True, True

    combined = np.concatenate(combined_parts)
    scaled_combined, valid = zscore_predictor_values(combined)
    if not valid:
        scaled_a = np.full_like(arr_a, np.nan, dtype=np.float64) if arr_a.size > 0 else arr_a.copy()
        scaled_b = np.full_like(arr_b, np.nan, dtype=np.float64) if arr_b.size > 0 else arr_b.copy()
        return scaled_a, scaled_b, arr_a.size == 0, arr_b.size == 0

    split = arr_a.size
    scaled_a = scaled_combined[:split].copy()
    scaled_b = scaled_combined[split:].copy()
    return scaled_a, scaled_b, True, True


def _zscore_predictor_or_empty(
    predictor_values: np.ndarray,
) -> tuple[np.ndarray, bool]:
    arr = np.asarray(predictor_values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy(), True
    scaled, valid = zscore_predictor_values(arr)
    if valid:
        return scaled, True
    return np.full_like(arr, np.nan, dtype=np.float64), False


def zscore_epochs_across_trials(
    epochs: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Z-score a ``(n_trials, n_features, n_times)`` array across the trial axis."""
    arr = np.asarray(epochs, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("epochs must be 3-D (n_trials, n_features, n_times).")
    if arr.shape[0] < 3:
        return np.full_like(arr, np.nan, dtype=np.float64), False

    mean = np.nanmean(arr, axis=0, dtype=np.float64)
    std = np.nanstd(arr, axis=0, ddof=1, dtype=np.float64)
    valid_std = np.isfinite(std) & (std > 0.0)
    if not np.any(valid_std):
        return np.full_like(arr, np.nan, dtype=np.float64), False

    out = np.full_like(arr, np.nan, dtype=np.float64)
    centered = arr - mean[np.newaxis, :, :]
    out[:, valid_std] = centered[:, valid_std] / std[valid_std]
    return out, True
