# Changelog

## 1.1.0 - 2026-06-26

### Added

- Added the trial-level `time_frequency` pipeline for FieldTrip-aligned multitaper power computation and BIDS derivative writing.
- Added subject-level time-frequency condition-test and regression pipelines, including HDF5/MATLAB-compatible result loaders and writers.
- Added group-level ROI statistics for time-frequency condition-test and regression outputs, with compatibility grouping helpers.
- Added public runner scripts and documentation for the time-frequency processing and statistics workflows.
- Added regression/statistics utilities shared by Hilbert/trial-statistics and time-frequency workflows.

### Changed

- Reworked parts of trial-statistics regression internals to share regression parameter and statistics helpers.
- Expanded documentation and README references for the new time-frequency workflows.
- Updated public snapshot tooling and release metadata for the 1.1.x release line.

### Fixed

- Fixed handling around significant-channel reporting in statistics outputs.
- Fixed assorted metadata and documentation issues found while preparing the time-frequency release.
