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

    # NaN-aware path: when per-channel NaN patterns are present (from trial-level
    # cleaning), each channel is regressed independently using only its valid trials.
    if not np.all(np.isfinite(y)):
        slope_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
        intercept_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
        r_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
        p_out = np.full((n_features, n_times), np.nan, dtype=np.float64)
        any_valid_channel = False
        for ch in range(n_features):
            # A trial is NaN for this channel if any time sample is non-finite;
            # apply_trial_nan_mask guarantees the mask is uniform across the time axis,
            # so checking timepoint 0 is sufficient.
            valid = np.isfinite(y[:, ch, 0])
            x_ch = x[valid]
            y_ch = y[valid, ch, :]  # [n_valid, n_times]
            n_ch = int(x_ch.shape[0])
            if n_ch < 3:
                continue
            var_x_ch = float(np.nanvar(x_ch, ddof=1))
            if not (np.isfinite(var_x_ch) and var_x_ch > 0.0):
                continue
            x_mean_ch = float(np.nanmean(x_ch))
            y_mean_ch = np.nanmean(y_ch, axis=0)  # [n_times]
            x_c = x_ch - x_mean_ch
            y_c = y_ch - y_mean_ch[np.newaxis, :]
            cov_ch = np.dot(x_c, y_c) / float(n_ch - 1)  # [n_times]
            slope_ch = cov_ch / var_x_ch
            intercept_ch = y_mean_ch - slope_ch * x_mean_ch
            var_y_ch = np.nanvar(y_ch, axis=0, ddof=1)
            denom_ch = np.sqrt(var_x_ch * var_y_ch)
            r_ch = np.full(n_times, np.nan, dtype=np.float64)
            corr_mask_ch = np.isfinite(denom_ch) & (denom_ch > 0.0)
            r_ch[corr_mask_ch] = np.clip(
                cov_ch[corr_mask_ch] / denom_ch[corr_mask_ch], -1.0, 1.0
            )
            df_ch = float(n_ch - 2)
            p_ch = np.full(n_times, np.nan, dtype=np.float64)
            valid_t_ch = np.isfinite(r_ch) & (np.abs(r_ch) < 1.0)
            if np.any(valid_t_ch):
                r_sq_ch = np.square(r_ch[valid_t_ch], dtype=np.float64)
                t_vals_ch = r_ch[valid_t_ch] * np.sqrt(df_ch / (1.0 - r_sq_ch))
                p_ch[valid_t_ch] = 2.0 * student_t.sf(np.abs(t_vals_ch), df_ch)
            perfect_ch = np.isfinite(r_ch) & (np.abs(r_ch) >= 1.0)
            p_ch[perfect_ch] = 0.0
            slope_out[ch] = slope_ch
            intercept_out[ch] = intercept_ch
            r_out[ch] = r_ch
            p_out[ch] = p_ch
            any_valid_channel = True
        return slope_out, intercept_out, r_out, p_out, any_valid_channel

    # Dense (no-NaN) path: vectorized over all features and time samples at once.
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


def compute_permuted_regression_maps(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
    n_features: int,
    n_times: int,
) -> np.ndarray:
    """Compute permuted slope maps by randomly shuffling the predictor across trials.

    Each iteration shuffles the predictor order (leaving the epoch data fixed),
    re-fits OLS, and records the resulting slope.  The dense (no-NaN) path reduces
    each permutation to a single matrix-vector product; the NaN-aware path falls
    back to per-channel loops.

    Parameters
    ----------
    predictor_values:
        Array of shape ``(n_trials,)``.
    epochs:
        Array of shape ``(n_trials, n_features, n_times)``.
    n_perm:
        Number of permutation iterations.
    rng:
        NumPy random Generator used for reproducible predictor shuffling.
    n_features, n_times:
        Expected output dimensions.

    Returns
    -------
    permuted_slopes : float32 array, shape ``(n_perm, n_features, n_times)``
        NaN-filled for conditions with fewer than 3 valid trials or zero predictor
        variance.  Returns an all-NaN array without raising.
    """
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

    # NaN-aware path: channels have heterogeneous valid-trial masks.
    if not np.all(np.isfinite(y)):
        for i in range(n_perm):
            perm_x = x[rng.permutation(n_trials)]
            x_mean_p = float(np.mean(perm_x))
            x_c_p = perm_x - x_mean_p
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

    # Dense path: vectorized over all features and times.
    x_mean = float(np.mean(x))
    y_flat = y.reshape(n_trials, -1)           # (n_trials, n_features * n_times)
    y_mean = np.mean(y_flat, axis=0)
    y_centered = y_flat - y_mean[np.newaxis, :]  # precomputed invariant

    for i in range(n_perm):
        perm_x = x[rng.permutation(n_trials)]
        x_c = perm_x - float(np.mean(perm_x))
        # cov = (x_c^T @ y_centered) / (n - 1)
        cov = np.dot(x_c, y_centered) / float(n_trials - 1)
        slope = (cov / var_x).reshape(n_features, n_times)
        out[i] = slope.astype(np.float32)

    return out


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
