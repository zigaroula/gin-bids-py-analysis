from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.stats import t as student_t

RegressionNanPolicy = Literal["feature_time0", "pointwise"]


def compute_linear_regression_maps(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_features: int,
    n_times: int,
    return_n_obs: bool = False,
    nan_policy: RegressionNanPolicy = "feature_time0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
]:
    """Compute per-feature, per-time OLS maps for y ~ x.

    ``nan_policy="feature_time0"`` preserves the historical trial_stats behavior:
    each feature uses the finite-trial mask observed at time index 0 for all time
    points. ``nan_policy="pointwise"`` matches Matlab ``glmfit``-style TF loops by
    recomputing the finite response rows independently for every feature/time point.
    """
    empty = np.full((n_features, n_times), np.nan, dtype=np.float64)
    empty_n = np.zeros((n_features, n_times), dtype=np.int64)

    x = np.asarray(predictor_values, dtype=np.float64).reshape(-1)
    y = np.asarray(epochs, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError("epochs must be 3-D (n_trials, n_features, n_times).")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"predictor/epochs mismatch: {x.shape[0]} predictor values for {y.shape[0]} epochs."
        )
    if nan_policy not in {"feature_time0", "pointwise"}:
        raise ValueError(
            f"Unsupported nan_policy: {nan_policy!r}. "
            "Valid values are 'feature_time0' and 'pointwise'."
        )

    finite_x = np.isfinite(x)
    if not np.all(finite_x):
        x = x[finite_x]
        y = y[finite_x]

    n_trials = int(x.shape[0])
    if n_trials < 3:
        return _linear_regression_return(
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            False,
            empty_n.copy(),
            return_n_obs=return_n_obs,
        )

    var_x = float(np.nanvar(x, ddof=1))
    if not np.isfinite(var_x) or var_x <= 0.0:
        return _linear_regression_return(
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            False,
            empty_n.copy(),
            return_n_obs=return_n_obs,
        )

    if not np.all(np.isfinite(y)):
        if nan_policy == "pointwise":
            return _compute_linear_regression_maps_pointwise_nan(
                x,
                y,
                n_features=n_features,
                n_times=n_times,
                return_n_obs=return_n_obs,
            )
        return _compute_linear_regression_maps_feature_time0_nan(
            x,
            y,
            n_features=n_features,
            n_times=n_times,
            return_n_obs=return_n_obs,
        )

    return _compute_linear_regression_maps_dense(
        x,
        y,
        n_features=n_features,
        n_times=n_times,
        return_n_obs=return_n_obs,
    )


def _compute_linear_regression_maps_feature_time0_nan(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int,
    n_times: int,
    return_n_obs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
]:
    slope_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    intercept_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    r_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    p_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    n_obs_out = np.zeros((n_features, n_times), dtype=np.int64)
    any_valid = False

    for ch in range(n_features):
        valid = np.isfinite(y[:, ch, 0])
        x_ch = x[valid]
        y_ch = y[valid, ch, :]
        n_ch = int(x_ch.shape[0])
        if n_ch < 3:
            continue

        var_x_ch = float(np.nanvar(x_ch, ddof=1))
        if not np.isfinite(var_x_ch) or var_x_ch <= 0.0:
            continue

        x_mean_ch = float(np.nanmean(x_ch))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y_mean_ch = np.nanmean(y_ch, axis=0, dtype=np.float64)
        x_c = x_ch - x_mean_ch
        y_c = y_ch - y_mean_ch[np.newaxis, :]
        cov_ch = np.dot(x_c, y_c) / float(n_ch - 1)

        slope = cov_ch / var_x_ch
        intercept = y_mean_ch - (slope * x_mean_ch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            var_y_ch = np.nanvar(y_ch, axis=0, ddof=1, dtype=np.float64)
        denom = np.sqrt(var_x_ch * var_y_ch)
        r = np.full(n_times, np.nan, dtype=np.float64)
        corr_mask = np.isfinite(denom) & (denom > 0.0)
        r[corr_mask] = cov_ch[corr_mask] / denom[corr_mask]
        r[corr_mask] = np.clip(r[corr_mask], -1.0, 1.0)

        p = np.full(n_times, np.nan, dtype=np.float64)
        df = float(n_ch - 2)
        valid_t = np.isfinite(r) & (np.abs(r) < 1.0)
        if np.any(valid_t):
            t_values = r[valid_t] * np.sqrt(df / (1.0 - np.square(r[valid_t], dtype=np.float64)))
            p[valid_t] = 2.0 * student_t.sf(np.abs(t_values), df)
        perfect = np.isfinite(r) & (np.abs(r) >= 1.0)
        p[perfect] = 0.0

        slope_out[ch, :] = slope
        intercept_out[ch, :] = intercept
        r_out[ch, :] = r
        p_out[ch, :] = p
        n_obs_out[ch, :] = n_ch
        any_valid = True

    return _linear_regression_return(
        slope_out,
        intercept_out,
        r_out,
        p_out,
        any_valid,
        n_obs_out,
        return_n_obs=return_n_obs,
    )


def _compute_linear_regression_maps_pointwise_nan(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int,
    n_times: int,
    return_n_obs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
]:
    slope_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    intercept_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    r_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    p_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
    n_obs_out = np.zeros((n_features, n_times), dtype=np.int64)
    any_valid = False
    chunk_size = 128
    x3 = x[:, np.newaxis, np.newaxis]

    for start in range(0, n_features, chunk_size):
        stop = min(start + chunk_size, n_features)
        y_chunk = y[:, start:stop, :]
        finite = np.isfinite(y_chunk)
        n_obs = np.sum(finite, axis=0, dtype=np.int64)
        valid_n = n_obs >= 3
        if not np.any(valid_n):
            continue

        n_float = n_obs.astype(np.float64)
        sum_x = np.sum(np.where(finite, x3, 0.0), axis=0, dtype=np.float64)
        sum_y = np.sum(np.where(finite, y_chunk, 0.0), axis=0, dtype=np.float64)
        mean_x = np.full_like(sum_x, np.nan, dtype=np.float64)
        mean_y = np.full_like(sum_y, np.nan, dtype=np.float64)
        np.divide(sum_x, n_float, out=mean_x, where=valid_n)
        np.divide(sum_y, n_float, out=mean_y, where=valid_n)

        dx = np.where(finite, x3 - mean_x[np.newaxis, :, :], 0.0)
        dy = np.where(finite, y_chunk - mean_y[np.newaxis, :, :], 0.0)
        ssx = np.sum(dx * dx, axis=0, dtype=np.float64)
        ssy = np.sum(dy * dy, axis=0, dtype=np.float64)
        sxy = np.sum(dx * dy, axis=0, dtype=np.float64)

        valid = valid_n & np.isfinite(ssx) & (ssx > 0.0) & np.isfinite(ssy) & (ssy > 0.0)
        if not np.any(valid):
            continue

        slope = np.full_like(sxy, np.nan, dtype=np.float64)
        r = np.full_like(sxy, np.nan, dtype=np.float64)
        np.divide(sxy, ssx, out=slope, where=valid)
        denom = np.sqrt(ssx * ssy)
        np.divide(sxy, denom, out=r, where=valid)
        r[valid] = np.clip(r[valid], -1.0, 1.0)
        intercept = mean_y - slope * mean_x

        p = np.full_like(r, np.nan, dtype=np.float64)
        df = n_float - 2.0
        valid_t = valid & np.isfinite(r) & (np.abs(r) < 1.0)
        if np.any(valid_t):
            r_sq = np.square(r[valid_t], dtype=np.float64)
            t_vals = r[valid_t] * np.sqrt(df[valid_t] / (1.0 - r_sq))
            p[valid_t] = 2.0 * student_t.sf(np.abs(t_vals), df[valid_t])
        perfect = valid & np.isfinite(r) & (np.abs(r) >= 1.0)
        p[perfect] = 0.0

        slope_out[start:stop] = slope
        intercept_out[start:stop] = intercept
        r_out[start:stop] = r
        p_out[start:stop] = p
        n_obs_out[start:stop] = np.where(valid, n_obs, 0)
        any_valid = True

    return _linear_regression_return(
        slope_out,
        intercept_out,
        r_out,
        p_out,
        any_valid,
        n_obs_out,
        return_n_obs=return_n_obs,
    )


def _compute_linear_regression_maps_dense(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_features: int,
    n_times: int,
    return_n_obs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
]:
    n_trials = int(x.shape[0])
    var_x = float(np.nanvar(x, ddof=1))
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

    return _linear_regression_return(
        slope_flat.reshape(n_features, n_times),
        intercept_flat.reshape(n_features, n_times),
        r_flat.reshape(n_features, n_times),
        p_flat.reshape(n_features, n_times),
        True,
        np.full((n_features, n_times), n_trials, dtype=np.int64),
        return_n_obs=return_n_obs,
    )


def _linear_regression_return(
    slope: np.ndarray,
    intercept: np.ndarray,
    r_value: np.ndarray,
    p_value: np.ndarray,
    valid: bool,
    n_obs: np.ndarray,
    *,
    return_n_obs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    np.ndarray,
]:
    if return_n_obs:
        return slope, intercept, r_value, p_value, valid, n_obs
    return slope, intercept, r_value, p_value, valid


def compute_permuted_regression_maps(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    """Compute permuted slope maps by randomly shuffling the predictor across trials."""
    empty = np.full((n_perm, n_features, n_times), np.nan, dtype=np.float32)

    if n_perm == 0:
        return empty

    x = np.asarray(predictor_values, dtype=np.float64).reshape(-1)
    y = np.asarray(epochs, dtype=np.float64)
    if y.ndim != 3:
        return empty

    finite_x = np.isfinite(x)
    if not np.all(finite_x):
        x = x[finite_x]
        y = y[finite_x]

    n_trials = int(x.shape[0])
    if n_trials < 3:
        return empty

    var_x = float(np.nanvar(x, ddof=1))
    if not np.isfinite(var_x) or var_x <= 0.0:
        return empty

    out = np.empty((n_perm, n_features, n_times), dtype=np.float32)

    if not np.all(np.isfinite(y)):
        for i in range(n_perm):
            perm_x = x[rng.permutation(n_trials)]
            for ch in range(n_features):
                valid = np.isfinite(y[:, ch, 0])
                x_ch = perm_x[valid]
                y_ch = y[valid, ch, :]
                n_ch = int(x_ch.shape[0])
                if n_ch < 3:
                    out[i, ch, :] = np.nan
                    continue
                var_x_ch = float(np.nanvar(x_ch, ddof=1))
                if not (np.isfinite(var_x_ch) and var_x_ch > 0.0):
                    out[i, ch, :] = np.nan
                    continue
                x_c_ch = x_ch - float(np.mean(x_ch))
                y_c_ch = y_ch - np.mean(y_ch, axis=0)
                cov_ch = np.dot(x_c_ch, y_c_ch) / float(n_ch - 1)
                out[i, ch, :] = (cov_ch / var_x_ch).astype(np.float32)
        return out

    y_flat = y.reshape(n_trials, -1)
    y_mean = np.mean(y_flat, axis=0)
    y_centered = y_flat - y_mean[np.newaxis, :]

    for i in range(n_perm):
        perm_x = x[rng.permutation(n_trials)]
        x_c = perm_x - float(np.mean(perm_x))
        cov = np.dot(x_c, y_centered) / float(n_trials - 1)
        slope = (cov / var_x).reshape(n_features, n_times)
        out[i] = slope.astype(np.float32)

    return out


def zscore_predictor_values(
    predictor_values: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Z-score a 1-D predictor across trials."""
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
    """Return predictor arrays z-scored with the requested scope."""
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
