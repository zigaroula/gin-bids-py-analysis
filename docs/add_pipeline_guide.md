# Add a Pipeline

This guide explains how to add a new `bidsforge` processing pipeline that follows the existing processor/writer pattern.

## 1. Choose the Pipeline Shape

Most pipelines in `bidsforge` use the same components:

```text
bidsforge/processing/<pipeline>/
|-- __init__.py
|-- params.py
|-- processor.py
|-- result.py
|-- writer.py
`-- result_loader.py        # optional, useful for downstream pipelines
```

Use this shape when the pipeline:

- has typed algorithm parameters;
- reads one `BIDSFile` or one `BIDSFileGroup` at a time;
- returns an in-memory result object;
- writes BIDS derivatives through a writer.

## 2. Define Parameters

Create a Pydantic model in `params.py`.

For a new standalone pipeline:

```python
from typing import Literal

from pydantic import Field

from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams


class MyPipelineParams(BaseProcessingParams):
    frequency_hz: float = Field(gt=0)
    mode: Literal["fast", "accurate"] = "accurate"


class MyPipelineWriterParams(BaseWriterParams):
    pipeline_label: str = "my_pipeline"
    output_modality: str = "ieeg"
    output_suffix: str = "stats"
    output_description: str = "mypipeline"
    output_format: Literal["hdf5", "matlab"] = "hdf5"
```

For trial-statistics variants, inherit the existing shared bases:

- `BaseTrialStatsParams` and `BaseTrialStatsWriterParams` for subject-level trial pipelines.
- `BaseTrialStatsGroupParams` and `BaseTrialStatsGroupWriterParams` for group-level pipelines.

Parameter rules:

- keep required scientific choices explicit;
- use `Field(...)` descriptions because they double as documentation;
- validate incompatible options with `@model_validator`;
- use stable string literals for output formats and modes.

## 3. Define the Result Object

Create a dataclass in `result.py` that extends `BaseProcessingResult`:

```python
from dataclasses import dataclass

import numpy as np

from bidsforge.processing.base import BaseProcessingResult


@dataclass
class MyPipelineResult(BaseProcessingResult):
    values: np.ndarray
    channel_names: list[str]
    sfreq: float
```

Keep the result object focused on data and provenance. Avoid file-writing behavior here.

## 4. Implement the Processor

Create a processor in `processor.py` by extending `BaseProcessing`:

```python
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.base import BaseProcessing

from .params import MyPipelineParams
from .result import MyPipelineResult


class MyPipelineProcessing(BaseProcessing):
    def __init__(self, params: MyPipelineParams) -> None:
        self.params = params

    def process_group(
        self,
        group: BIDSFileGroup,
        progress_tracking_position: int = 0,
    ) -> MyPipelineResult:
        del progress_tracking_position
        with group.primary.ensure_loaded() as raw:
            data = raw.get_data()
            sfreq = float(raw.info["sfreq"])
            channel_names = list(raw.ch_names)

        values = run_algorithm(data, sfreq=sfreq, params=self.params)
        return MyPipelineResult(
            source_group=group,
            values=values,
            channel_names=channel_names,
            sfreq=sfreq,
            metadata=self.params.model_dump(),
        )
```

Processor rules:

- keep the processor stateless between groups;
- let `BaseProcessing.run` handle parallelism, skipping, and dataset descriptions;
- raise clear `ValueError` messages for invalid inputs;
- use existing helpers from `bidsforge.processing.utils` before adding new generic utilities.

## 5. Implement the Writer

Create a writer in `writer.py` by extending `BaseProcessingWriter`:

```python
import h5py

from bidsforge.processing.base import BaseProcessingWriter

from .params import MyPipelineWriterParams
from .result import MyPipelineResult


class MyPipelineProcessingWriter(BaseProcessingWriter):
    def __init__(self, params: MyPipelineWriterParams) -> None:
        super().__init__(params)

    def _write_data(self, result: MyPipelineResult, output_path):
        if self.params.output_format == "hdf5":
            with h5py.File(output_path, "w") as h5:
                h5.create_dataset("values", data=result.values)
                h5.attrs["schema_name"] = "my_pipeline"
                h5.attrs["sfreq"] = result.sfreq
        else:
            raise ValueError(f"Unsupported output_format: {self.params.output_format}")
```

Writer rules:

- do not build output paths manually unless the default BIDS path is insufficient;
- store enough metadata to reload and audit the result;
- keep schema names/version fields stable if downstream code will read the file;
- add a `result_loader.py` if another pipeline will consume the output.

## 6. Export the Public API

In the pipeline package `__init__.py`, export the public classes:

```python
from .params import MyPipelineParams, MyPipelineWriterParams
from .processor import MyPipelineProcessing
from .result import MyPipelineResult
from .writer import MyPipelineProcessingWriter

__all__ = [
    "MyPipelineParams",
    "MyPipelineWriterParams",
    "MyPipelineProcessing",
    "MyPipelineResult",
    "MyPipelineProcessingWriter",
]
```

If the pipeline belongs in a shared namespace, also update the parent package `__init__.py`.

## 7. Add a Script Recipe

Add an editable script under `scripts/` only if it helps users run the pipeline:

```python
from pathlib import Path

from bidsforge.bids import BIDSDataset
from bidsforge.processing.my_pipeline import (
    MyPipelineParams,
    MyPipelineProcessing,
    MyPipelineProcessingWriter,
    MyPipelineWriterParams,
)

BIDS_ROOT = Path("D:/data/bids")
FILTERS = {"scope": "raw", "suffix": "ieeg", "extension": ".vhdr"}

ds = BIDSDataset(BIDS_ROOT)
groups = ds.get_files(**FILTERS)

processor = MyPipelineProcessing(MyPipelineParams(frequency_hz=100.0))
writer = MyPipelineProcessingWriter(MyPipelineWriterParams(bids_root=BIDS_ROOT))
processor.run(groups, writer, n_jobs=1, skip_existing=True)
```

Scripts should be easy to edit and explicit about filters and parameters.

## 8. Add Tests

Add tests under `tests/processing/<pipeline>/`.

Recommended coverage:

- parameter validation;
- processor behavior on a minimal synthetic input;
- writer output path and file contents;
- loader round-trip if a loader exists;
- error messages for missing required sidecars or incompatible inputs.

For group or trial pipelines, also test compatibility grouping and audit metadata.

## 9. Add Documentation

Add `docs/pipelines/<pipeline>.md` with:

- objective;
- processing summary;
- minimal example;
- algorithm parameters;
- writer parameters;
- outputs;
- compatibility or downstream notes.

Reference the page from [docs/README.md](./README.md) and the root [README](../README.md) only when the pipeline is part of the public release.

## 10. Final Checklist

- Parameters are typed and validated.
- Processor is stateless and uses `BIDSFileGroup`.
- Writer uses `BaseProcessingWriter` and writes provenance-rich derivatives.
- Outputs land under `derivatives/<pipeline_label>/`.
- Tests cover params, processing, writing, and failure paths.
- Documentation includes both algorithm and writer parameters.
- Public exports are available from the intended package.
