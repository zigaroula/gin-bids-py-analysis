from .file import BIDSFile
from .file_group import BIDSFileGroup
from .helpers import build_bids_path, build_subject_groups, parse_entities
from .subject import BIDSSubject

try:
    from .dataset import BIDSDataset
except ImportError as exc:  # pragma: no cover - exercised only without pybids installed
    _dataset_import_error = exc

    class BIDSDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "BIDSDataset requires the optional 'pybids' dependency to be installed."
            ) from _dataset_import_error

__all__ = [
    "BIDSDataset",
    "BIDSFile",
    "BIDSFileGroup",
    "BIDSSubject",
    "build_bids_path",
    "build_subject_groups",
    "parse_entities",
]
