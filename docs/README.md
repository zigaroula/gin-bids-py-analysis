# Documentation Index

This directory has three documentation layers. Start with the end-to-end guide if you want to run an analysis; use pipeline pages when you need parameter-level detail; use the developer guide when extending the codebase.

## User Guides

- [End-to-End Guide](./end_to_end_guide.md): install, select inputs, run pipelines, inspect outputs.
- [Troubleshooting Guide](./troubleshooting_guide.md): common installation, BIDS discovery, processing, and writer failures.

## Pipeline Guides

- [Hilbert Pipeline](./pipelines/hilbert.md): band envelope extraction from iEEG recordings.
- [Condition Test Pipeline](./pipelines/condition_test.md): subject-level two-condition statistics.
- [Regression Pipeline](./pipelines/regression.md): subject-level condition-specific slope regression.
- [Condition Test Group Pipeline](./pipelines/condition_test_group.md): group-level ROI statistics from `condition_test` outputs.
- [Regression Group Pipeline](./pipelines/regression_group.md): group-level ROI statistics from `regression` outputs.

## Developer Guides

- [Add a Pipeline](./add_pipeline_guide.md): implement parameters, processor, result, writer, tests, and docs for a new pipeline.

## Recommended Reading Paths

- New user: README -> End-to-End Guide -> relevant pipeline guide.
- Subject-level statistics: End-to-End Guide -> Condition Test or Regression Pipeline.
- Group analysis: subject-level pipeline guide -> matching group pipeline guide.
- New contributor: Add a Pipeline -> nearby existing pipeline implementation -> tests.

## About `scripts/`

Repository-level scripts can contain editable analysis recipes for real projects and quick inspection utilities. They are useful starting points, but the supported Python API is the pipeline pattern documented in the end-to-end and pipeline guides.
