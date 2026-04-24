"""Debug X02X01 slope difference: compare z-score reference stats between
the regression pipeline (174-trial pool) and MATLAB b2 (240-trial pool).
"""
from __future__ import annotations
import csv, sys, warnings
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_subject_config import MATLAB_B2_PATH, BEHAVIOR_TSV_PATH
from compare_b3 import _load_matlab_b2_hdf5, compute_python_regression_from_source

import re

TARGET_SUB = "REN2021SIDs"
TARGET_CH = "X02"   # Python bipolar name
TARGET_CH_MAT = "X02X01"  # MATLAB bipolar name

print(f"=== X02X01 z-score reference pool analysis ===")

# ---------------------------------------------------------------------------
# 1. Python regression (174 trials: 90 pleasant + 84 unpleasant)
# ---------------------------------------------------------------------------
print("\n--- Python regression (174-trial baseline pool) ---")
fresh = compute_python_regression_from_source(target_subject=TARGET_SUB)

py_chs = list(fresh.channel_names)
x02_py = next((i for i, c in enumerate(py_chs) if c.strip().upper() == TARGET_CH.upper()), None)
if x02_py is None:
    print(f"  {TARGET_CH} not found in Python channels: {py_chs[:10]}")
    sys.exit(1)

time_axis = np.asarray(fresh.time_axis_s, dtype=np.float64)
bsl_mask = (time_axis >= -0.25) & (time_axis <= -0.06)
n_bsl_py = int(np.sum(bsl_mask))
print(f"  {TARGET_CH} at Python channel index {x02_py}")
print(f"  Baseline window: {float(time_axis[bsl_mask][0]):.4f} → {float(time_axis[bsl_mask][-1]):.4f} s ({n_bsl_py} samples)")

epochs_a = np.asarray(fresh.condition_a_epochs, dtype=np.float64)
epochs_b = np.asarray(fresh.condition_b_epochs, dtype=np.float64)

x02_a = epochs_a[:, x02_py, :]  # (90, T) — already z-scored
x02_b = epochs_b[:, x02_py, :]  # (84, T)
x02_pool = np.concatenate([x02_a, x02_b], axis=0)  # (174, T)

# Per-trial baseline means of the ALREADY Z-SCORED epochs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bsl_means_pool = np.nanmean(x02_pool[:, bsl_mask], axis=1)   # (174,)
    bsl_means_a = np.nanmean(x02_a[:, bsl_mask], axis=1)          # (90,)

n_nan_a = int(np.sum(np.all(~np.isfinite(x02_a), axis=1)))
n_nan_b = int(np.sum(np.all(~np.isfinite(x02_b), axis=1)))
n_nan_pool = int(np.sum(np.all(~np.isfinite(x02_pool), axis=1)))

print(f"  Condition A (90 trials, {n_nan_a} fully NaN):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"    per-trial bsl mean: mean={float(np.nanmean(bsl_means_a)):.5f}, std={float(np.nanstd(bsl_means_a)):.5f}")
print(f"  Pool (174 trials, {n_nan_pool} fully NaN):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"    per-trial bsl mean: mean={float(np.nanmean(bsl_means_pool)):.5f}, std={float(np.nanstd(bsl_means_pool)):.5f}")
print(f"  (Values in z-score space — should be ~0 if reference was correct)")

# To recover the original reference: the z-scoring uses the Python pre-zscore data.
# We can verify by checking whether the raw Hilbert epochs for X02X01 (before zscore)
# have the same PRECLEAN mask as MATLAB.
# Instead, let's look at the effective reference by checking baseline std.
# In z-score space: mean_bsl ≈ (raw_bsl_mean - ref_mean) / ref_std
# So if all trials' bsl_mean ≈ 0, ref_mean ≈ mean(raw_bsl_means)
# and bsl_std (z) ≈ std(raw_bsl_means) / ref_std = 1 when ref_std = std(raw_bsl_means)

# But since we have NaN masks, the reference was computed AFTER removing outliers.
# The per-trial baseline means in z-score space tell us the spread of
# (raw_bsl_mean_i - ref_mean) / ref_std, which should be ~N(0,1) for most trials.
print(f"  Note: spread should be ~1.0 if the z-scoring normalized correctly.")

# ---------------------------------------------------------------------------
# 2. MATLAB b2 (240 trials, with NaN per channel from PRECLEAN)
# ---------------------------------------------------------------------------
print("\n--- MATLAB b2 (240-trial baseline pool) ---")
b2_alldata, b2_channels, b2_timelist = _load_matlab_b2_hdf5(MATLAB_B2_PATH)
print(f"  alldata shape: {b2_alldata.shape}  (trials, time, channels)")

def _first_contact(name: str) -> str:
    m = re.match(r"^([A-Za-z]+\d+)", str(name))
    return m.group(1).upper() if m else str(name).upper()

b2_norm = [_first_contact(c) for c in b2_channels]
x02_mat = next((i for i, c in enumerate(b2_norm) if c == TARGET_CH_MAT.split("X")[0] + "02" or c == "X02"), None)
# More flexible match
x02_mat = next((i for i, c in enumerate(b2_channels) if str(c).upper() == TARGET_CH_MAT.upper()), None)
if x02_mat is None:
    print(f"  {TARGET_CH_MAT!r} not found. Available: {b2_channels[:20]}")
    sys.exit(1)

print(f"  {TARGET_CH_MAT} at MATLAB index {x02_mat} ('{b2_channels[x02_mat]}')")
bsl_mask_mat = (b2_timelist >= -0.25) & (b2_timelist <= -0.06)
print(f"  Baseline: {int(np.sum(bsl_mask_mat))} samples "
      f"({float(b2_timelist[bsl_mask_mat][0]):.4f} → {float(b2_timelist[bsl_mask_mat][-1]):.4f} s)")

# Load pleasantness
pleasantness: list[int] = []
with open(BEHAVIOR_TSV_PATH, "r", encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        pleasantness.append(int(float(row["pleasant"])))
pleasant_mask = np.array(pleasantness) == 1
unpleasant_mask = np.array(pleasantness) == 2

x02_all = b2_alldata[:, :, x02_mat]  # (240, T_mat) — already z-scored by MATLAB b2
x02_pleasant = x02_all[pleasant_mask]   # (120, T_mat)
x02_unpleasant = x02_all[unpleasant_mask]  # (120, T_mat)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bsl_means_all = np.nanmean(x02_all[:, bsl_mask_mat], axis=1)  # (240,)
    bsl_means_pleasant = np.nanmean(x02_pleasant[:, bsl_mask_mat], axis=1)  # (120,)

n_nan_mat_all = int(np.sum(np.all(~np.isfinite(x02_all), axis=1)))
n_nan_mat_pls = int(np.sum(np.all(~np.isfinite(x02_pleasant), axis=1)))
n_nan_mat_unp = int(np.sum(np.all(~np.isfinite(x02_unpleasant), axis=1)))

print(f"  All 240 trials ({n_nan_mat_all} fully NaN):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"    per-trial bsl mean: mean={float(np.nanmean(bsl_means_all)):.5f}, std={float(np.nanstd(bsl_means_all)):.5f}")
print(f"  Pleasant 120 trials ({n_nan_mat_pls} fully NaN):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"    per-trial bsl mean: mean={float(np.nanmean(bsl_means_pleasant)):.5f}, std={float(np.nanstd(bsl_means_pleasant)):.5f}")

# ---------------------------------------------------------------------------
# 3. Side-by-side: MATLAB b2 z-scored vs Python regression z-scored for 90 trials
# ---------------------------------------------------------------------------
print("\n--- Comparing MATLAB b2 vs Python regression z-scored epochs for X02X01 (90 pleasant trials) ---")
print("  This requires mapping MATLAB b2 trial indices to Python trial indices.")

# Python condition_a has 90 trials with finite matlab_zscore.
# MATLAB has 120 pleasant trials; 90 have finite P_Rating.
# We need to know which MATLAB b2 pleasant trial indices correspond to the 90 finite ones.
# Those are the trials for which MATLAB b3 P_Rating regressor is finite.

# Load MATLAB b3 P_Rating regressor to get finite trial indices
from compare_subject_config import MATLAB_B3_ROOT
from compare_b3 import load_matlab_b3_regressor

b3_log_path = MATLAB_B3_ROOT / "P_Rating" / "log_data.mat"
mat_reg = load_matlab_b3_regressor(b3_log_path, regression_name="P_Rating")
# mat_reg has 120 entries, one per pleasant trial; 90 are finite
mat_finite_mask = np.isfinite(mat_reg)  # (120,) bool
print(f"  MATLAB b3 P_Rating: {int(np.sum(mat_finite_mask))} finite out of {len(mat_reg)} pleasant trials")

# X02X01: get 90 finite MATLAB pleasant trials
x02_mat_90 = x02_pleasant[mat_finite_mask]  # (90, T_mat)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bsl_means_mat_90 = np.nanmean(x02_mat_90[:, bsl_mask_mat], axis=1)  # (90,)
n_nan_mat_90 = int(np.sum(np.all(~np.isfinite(x02_mat_90), axis=1)))
print(f"  MATLAB b2, X02X01, 90 finite-predictor pleasant trials ({n_nan_mat_90} fully NaN):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"    bsl mean: mean={float(np.nanmean(bsl_means_mat_90)):.5f}, std={float(np.nanstd(bsl_means_mat_90)):.5f}")

# Python condition_a, X02X01, 90 trials (z-scored with 174-trial pool):
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"  Python regression, X02, 90 condition_a trials ({n_nan_a} fully NaN):")
    print(f"    bsl mean: mean={float(np.nanmean(bsl_means_a)):.5f}, std={float(np.nanstd(bsl_means_a)):.5f}")

# Direct value comparison: MATLAB b2 vs Python for each of 90 trials
# Use a subset of time points for comparison
n_times_mat = x02_mat_90.shape[1]
n_times_py = x02_a.shape[1]

if n_times_mat == n_times_py:
    diff_90 = x02_a - x02_mat_90
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"\n  Direct value comparison (90 trials, {n_times_py} time points each):")
        print(f"    max|diff| = {float(np.nanmax(np.abs(diff_90))):.5f}")
        print(f"    mean|diff| = {float(np.nanmean(np.abs(diff_90))):.5f}")
else:
    # Time axes may differ (MATLAB 599 pts vs Python 701 pts) — compare only shared window
    print(f"\n  Time axis mismatch: MATLAB {n_times_mat} pts, Python {n_times_py} pts")
    # Find common time window
    tmin_common = max(float(b2_timelist[0]), float(time_axis[0]))
    tmax_common = min(float(b2_timelist[-1]), float(time_axis[-1]))
    shared_mat = (b2_timelist >= tmin_common) & (b2_timelist <= tmax_common)
    shared_py = (time_axis >= tmin_common) & (time_axis <= tmax_common)
    print(f"  Shared window: {tmin_common:.3f} → {tmax_common:.3f} s  "
          f"(MATLAB: {int(np.sum(shared_mat))} pts, Python: {int(np.sum(shared_py))} pts)")
    
    if int(np.sum(shared_mat)) == int(np.sum(shared_py)):
        x02_mat_shared = x02_mat_90[:, shared_mat]  # (90, T_shared)
        x02_py_shared = x02_a[:, shared_py]         # (90, T_shared)
        diff_90 = x02_py_shared - x02_mat_shared
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"  Direct value comparison (90 trials, shared {int(np.sum(shared_mat))} time points):")
            print(f"    max|diff| = {float(np.nanmax(np.abs(diff_90))):.5f}")
            print(f"    mean|diff| = {float(np.nanmean(np.abs(diff_90))):.5f}")
            print(f"    per-trial mean|diff|: min={float(np.nanmin(np.nanmean(np.abs(diff_90), axis=1))):.5f}, "
                  f"max={float(np.nanmax(np.nanmean(np.abs(diff_90), axis=1))):.5f}")
    else:
        print(f"  Shared time points don't match — skipping direct comparison")

# ---------------------------------------------------------------------------
# 4. Simulate "correct" z-score: what if Python also used 240 trials?
# ---------------------------------------------------------------------------
print("\n--- Simulation: what if Python z-scored with 240-trial pool? ---")
print("  Computing Python z-score reference from 240 trials is not straightforward")
print("  without re-running the full pipeline. Instead:")
print("  → The key question is: do the MATLAB b2 baseline stats (from 240 trials) differ")
print("    significantly from Python's (174 trials) for X02X01?")

# MATLAB: from baseline means of all 240 trials (after NaN removal by PRECLEAN)
# The z-scoring reference is computed BEFORE MATLAB does any per-channel outlier removal.
# Actually, MATLAB b2 z-scoring uses ALL non-NaN trials. Let me estimate:
finite_mat_bsl_means = bsl_means_all[np.isfinite(bsl_means_all)]
finite_py_bsl_means = bsl_means_pool[np.isfinite(bsl_means_pool)]
print(f"\n  MATLAB b2 all-240-trial bsl means (in z-score space): "
      f"n_finite={len(finite_mat_bsl_means)}, "
      f"mean={float(np.mean(finite_mat_bsl_means)):.5f}, "
      f"std={float(np.std(finite_mat_bsl_means)):.5f}")
print(f"  Python 174-trial bsl means (in z-score space): "
      f"n_finite={len(finite_py_bsl_means)}, "
      f"mean={float(np.mean(finite_py_bsl_means)):.5f}, "
      f"std={float(np.std(finite_py_bsl_means)):.5f}")
print()
print("  If the z-score spaces match (both ≈ 0 mean, ≈ 1 std), the data is consistent.")
print("  Significant differences would indicate the z-score reference pools differ for X02X01.")
