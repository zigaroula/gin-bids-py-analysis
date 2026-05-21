"""Shared writer helpers for group-level trial statistics."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from gin_bids_py_analysis.processing.base import BaseProcessingWriter


def package_version() -> str:
    """Return the installed package version when available."""

    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class BaseTrialStatsGroupProcessingWriter(BaseProcessingWriter):
    """Shared helper methods for group-level result writers."""

    def _build_output_path(self, entities: dict) -> Path:
        """Force ``subject="group"`` before delegating to the base path builder."""
        group_entities = dict(entities)
        group_entities["subject"] = "group"
        return super()._build_output_path(group_entities)
