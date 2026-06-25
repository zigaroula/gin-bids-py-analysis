from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFileGroup
from bidsforge.processing.hilbert import HilbertParams, HilbertProcessing, HilbertProcessingWriter, HilbertWriterParams

BIDS_ROOT = Path(r"E:\ebrains\bids")
IEEG_FILTERS = {"scope": "raw", "datatype": "ieeg", "suffix": "ieeg", "extension": ".vhdr"}

ds = BIDSDataset(BIDS_ROOT)
groups = [BIDSFileGroup(primary=f) for f in sorted(ds.get_files(**IEEG_FILTERS), key=lambda x: str(x.path))]

params = HilbertParams(f_min=50, f_max=150, f_step=10, downsampled_frequency_hz=64, method="localizer", montage_mode="bipolar", bipolar_storage="next")
writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=BIDS_ROOT, output_description="gamma", output_format="brainvision"))

print(f"Found {len(groups)} iEEG file(s).")
for path in HilbertProcessing(params).run(groups, writer, n_jobs=1, skip_existing=True):
    print(path)
