# Development Scripts

This directory contains private maintenance tools for repository owners. It is
not included in public snapshots.

## Public Snapshot Workflow

`create_public_snapshot.py` creates or updates a separate local Git repository
containing only files approved for public release. It does not publish this
repository's history.

The public snapshot currently includes:

- `.gitignore`, generated from `scripts/dev/public.gitignore`
- `README.md`
- `pyproject.toml`
- `bidsforge/`
- `docs/`
- public recipe scripts in `scripts/*.py`
- `scripts/README.md`
- optional release files such as `LICENSE`, `LICENSE.md`, `CHANGELOG.md`, and
  `CITATION.cff`

The public snapshot currently excludes:

- `bidsforge/processing/hfo_spike_detection/`
- `docs/pipelines/hfo_spike_detection.md`
- `scripts/dev/`
- `.github/`
- local caches, virtual environments, build artifacts, and generated outputs

Edit `ALLOWLIST`, `OPTIONAL_ALLOWLIST`, and `DENYLIST` near the top of
`create_public_snapshot.py` when the public file selection changes.

## First Setup

The default destination is a sibling directory:

```bash
../bidsforge-public
```

Relative paths passed to `--output` are resolved from the repository root, not
from `scripts/dev/`.

If the public repository already exists remotely, clone it first:

```bash
cd ..
git clone git@github.com:GIN/bidsforge.git bidsforge-public
cd bidsforge
```

If the public repository has not been cloned yet, the script can initialize the
local repository automatically:

```bash
python scripts/dev/create_public_snapshot.py --output ../bidsforge-public
```

## Preview The Snapshot

From this repository root:

```bash
python scripts/dev/create_public_snapshot.py --dry-run
```

Review the selected file list carefully before the first public sync, and any
time the allowlist changes.

## Sync Without Committing

```bash
python scripts/dev/create_public_snapshot.py --output ../bidsforge-public
```

The script keeps the destination `.git/` directory intact, removes previously
copied public files, copies the current allowlisted files, writes `.gitignore`
from `scripts/dev/public.gitignore`, runs safety checks, and reports the
result.

## Create A Public Commit

After the checks pass, create a public release commit:

```bash
python scripts/dev/create_public_snapshot.py --commit --message "Release 1.0.0"
```

To also create an annotated tag:

```bash
python scripts/dev/create_public_snapshot.py --commit --message "Release 1.0.0" --tag v1.0.0
```

Then push from the public repository:

```bash
cd ../bidsforge-public
git push origin master
git push origin v1.0.0
cd ../bidsforge
```

Use `--skip-tests` only for local diagnostics. External scanners `gitleaks` and
`trufflehog` are used automatically when installed.
