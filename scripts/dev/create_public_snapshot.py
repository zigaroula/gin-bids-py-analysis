"""Create a sanitized public snapshot repository for bidsforge.

This script copies only allowlisted files from this working tree into a
separate local Git repository. The destination repository then receives one
commit per public release/snapshot, without exposing this repository's history.

Edit ALLOWLIST, OPTIONAL_ALLOWLIST, and DENYLIST below to choose exactly what
gets published.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT = "../bidsforge-public"
PUBLIC_GITIGNORE_TEMPLATE = Path("scripts/dev/public.gitignore")

ALLOWLIST = [
    ".gitignore",
    "README.md",
    "codemeta.json",
    "pyproject.toml",
    "bidsforge/**",
    "docs/**",
    "scripts/**",
    "tests/**",
]

OPTIONAL_ALLOWLIST = [
    "LICENSE",
    "LICENSE.md",
    "CHANGELOG.md",
    "CITATION.cff",
]

DENYLIST = [
    ".coverage",
    ".git/**",
    ".github/**",
    ".pytest_cache/**",
    ".test_tmp/**",
    ".venv/**",
    ".vscode/**",
    "__pycache__/**",
    "*.egg-info/**",
    "*.pyc",
    "build/**",
    "dist/**",
    "outputs/**",
    "scripts/dev/**",
    "scripts/debug/**",
    "docs/pipelines/hfo_spike_detection.md",
    "bidsforge/processing/hfo_spike_detection/**",
    "tests/processing/test_hfo_spike_detector_*.py",
]

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*"
        r"(?:['\"][^'\"]{12,}['\"]|[A-Za-z0-9_\-]{20,})"
    ),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def normalize(path: Path) -> str:
    return path.as_posix()


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_allowlisted(path: str) -> bool:
    return matches_any(path, ALLOWLIST) or matches_any(path, OPTIONAL_ALLOWLIST)


def is_denied(path: str) -> bool:
    parts = path.split("/")
    if any(part == "__pycache__" for part in parts):
        return True
    return matches_any(path, DENYLIST)


def iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = normalize(path.relative_to(root))
        if is_allowlisted(rel) and not is_denied(rel):
            files.append(path)
    return sorted(files)


def ensure_not_inside_source(source_root: Path, output_root: Path) -> None:
    try:
        output_root.relative_to(source_root)
    except ValueError:
        return
    raise SystemExit(
        f"Refusing to write the public repository inside the source repository: {output_root}"
    )


def clean_destination(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for child in output_root.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_snapshot(source_root: Path, output_root: Path, files: list[Path]) -> None:
    for src in files:
        rel = src.relative_to(source_root)
        dest = output_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def copy_public_gitignore(source_root: Path, output_root: Path) -> None:
    template = source_root / PUBLIC_GITIGNORE_TEMPLATE
    if not template.exists():
        raise SystemExit(f"Missing public .gitignore template: {template}")
    shutil.copy2(template, output_root / ".gitignore")


def run(
    command: list[str],
    cwd: Path,
    *,
    check: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ensure_git_repo(output_root: Path, branch: str) -> None:
    if not (output_root / ".git").exists():
        run(["git", "init", "-b", branch], output_root, check=True, capture=False)


def check_allowlist(output_root: Path) -> CheckResult:
    violations: list[str] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        rel = normalize(path.relative_to(output_root))
        if rel.startswith(".git/"):
            continue
        if not is_allowlisted(rel) or is_denied(rel):
            violations.append(rel)
    if violations:
        detail = "Unexpected files: " + ", ".join(violations[:20])
        if len(violations) > 20:
            detail += f" (+{len(violations) - 20} more)"
        return CheckResult("allowlist", False, detail)
    return CheckResult("allowlist", True, "Only allowlisted files are present.")


def check_forbidden_paths(output_root: Path) -> CheckResult:
    forbidden = []
    for path in output_root.rglob("*"):
        rel = normalize(path.relative_to(output_root))
        if rel.startswith(".git/"):
            continue
        if is_denied(rel):
            forbidden.append(rel)
    if forbidden:
        return CheckResult("forbidden paths", False, ", ".join(forbidden[:20]))
    return CheckResult("forbidden paths", True, "No forbidden paths are present.")


def scan_file_for_secrets(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    matches = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                matches.append(f"{path}:{line_number}")
                break
    return matches


def check_internal_secret_scan(output_root: Path) -> CheckResult:
    matches: list[str] = []
    for path in output_root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            matches.extend(scan_file_for_secrets(path))
    if matches:
        return CheckResult("internal secret scan", False, ", ".join(matches[:20]))
    return CheckResult("internal secret scan", True, "No obvious secrets found.")


def check_external_scanner(output_root: Path, executable: str) -> CheckResult:
    if shutil.which(executable) is None:
        return CheckResult(executable, True, "Not installed; skipped.")

    if executable == "gitleaks":
        command = [
            "gitleaks",
            "detect",
            "--no-git",
            "--source",
            str(output_root),
            "--redact",
            "--verbose",
        ]
    elif executable == "trufflehog":
        command = [
            "trufflehog",
            "filesystem",
            str(output_root),
            "--no-update",
            "--fail",
        ]
    else:
        raise ValueError(executable)

    result = run(command, output_root, capture=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return CheckResult(executable, False, detail[-2000:] or "Scanner failed.")
    return CheckResult(executable, True, "No findings.")


def check_tests(output_root: Path, skip: bool) -> CheckResult:
    if skip:
        return CheckResult("tests", True, "Skipped by request.")
    if not (output_root / "tests").exists():
        return CheckResult("tests", True, "No tests directory in snapshot.")

    result = run([sys.executable, "-m", "pytest"], output_root, capture=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return CheckResult("tests", False, detail[-4000:] or "pytest failed.")
    return CheckResult("tests", True, "pytest passed.")


def cleanup_generated_artifacts(output_root: Path) -> CheckResult:
    paths = [
        output_root / ".coverage",
        output_root / ".pytest_cache",
        output_root / ".test_tmp",
        output_root / "build",
        output_root / "dist",
    ]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for egg_info in output_root.glob("*.egg-info"):
        if egg_info.is_dir():
            shutil.rmtree(egg_info)
    for pycache in output_root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
    return CheckResult("generated artifacts", True, "Generated test/build artifacts removed.")


def run_checks(output_root: Path, *, skip_tests: bool) -> list[CheckResult]:
    return [
        check_internal_secret_scan(output_root),
        check_external_scanner(output_root, "gitleaks"),
        check_external_scanner(output_root, "trufflehog"),
        check_tests(output_root, skip_tests),
        cleanup_generated_artifacts(output_root),
        check_allowlist(output_root),
        check_forbidden_paths(output_root),
    ]


def print_checks(results: list[CheckResult]) -> None:
    for result in results:
        marker = "OK" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")


def fail_if_needed(results: list[CheckResult]) -> None:
    failed = [result for result in results if not result.ok]
    if failed:
        raise SystemExit("Snapshot checks failed; no public commit was created.")


def has_changes(output_root: Path) -> bool:
    result = run(["git", "status", "--porcelain"], output_root, check=True)
    return bool(result.stdout.strip())


def commit_snapshot(output_root: Path, message: str, tag: str | None) -> None:
    run(["git", "add", "-A"], output_root, check=True)
    if not has_changes(output_root):
        print("No public changes to commit.")
        return
    run(["git", "commit", "-m", message], output_root, check=True, capture=False)
    if tag:
        run(["git", "tag", "-a", tag, "-m", tag], output_root, check=True, capture=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sanitized public snapshot repository for bidsforge."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Destination public repository path. Relative paths are resolved from the "
            f"source repository root. Default: {DEFAULT_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--branch",
        default="master",
        help="Branch name to use when initializing the public repository.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit the snapshot in the destination repository after checks pass.",
    )
    parser.add_argument(
        "--message",
        default="Public snapshot",
        help="Commit message used with --commit.",
    )
    parser.add_argument(
        "--tag",
        help="Optional annotated Git tag to create after committing.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest in the public snapshot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without modifying the destination.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = repo_root()
    output_root = Path(args.output).expanduser()
    if not output_root.is_absolute():
        output_root = (source_root / output_root).resolve()
    else:
        output_root = output_root.resolve()

    ensure_not_inside_source(source_root, output_root)
    files = iter_source_files(source_root)

    print(f"Source: {source_root}")
    print(f"Destination: {output_root}")
    print(f"Files selected: {len(files)}")

    if args.dry_run:
        for src in files:
            print(normalize(src.relative_to(source_root)))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_git_repo(output_root, args.branch)
    clean_destination(output_root)
    copy_snapshot(source_root, output_root, files)
    copy_public_gitignore(source_root, output_root)

    results = run_checks(output_root, skip_tests=args.skip_tests)
    print_checks(results)
    fail_if_needed(results)

    if args.commit:
        commit_snapshot(output_root, args.message, args.tag)
    else:
        print("Checks passed. Re-run with --commit to create a public Git commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
