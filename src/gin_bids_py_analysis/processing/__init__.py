try:
    from .base import BaseProcessing, BaseProcessingResult, BaseProcessingWriter, BaseWriterParams
except ImportError as exc:  # pragma: no cover - exercised only in minimal environments
    _processing_import_error = exc

    class _MissingProcessingDependency:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "Processing base classes require optional runtime dependencies to be installed."
            ) from _processing_import_error

    BaseProcessing = _MissingProcessingDependency  # type: ignore[assignment]
    BaseProcessingResult = _MissingProcessingDependency  # type: ignore[assignment]
    BaseProcessingWriter = _MissingProcessingDependency  # type: ignore[assignment]
    BaseWriterParams = _MissingProcessingDependency  # type: ignore[assignment]

__all__ = [
    "BaseProcessing",
    "BaseProcessingResult",
    "BaseProcessingWriter",
    "BaseWriterParams",
]
