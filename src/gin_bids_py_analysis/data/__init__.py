# TODO: This module is a placeholder.
#
#       The iEEG data loading code will be copy-pasted from the source project
#       and later migrated to a proper git submodule.
#
#       Source repository: <REPO_URL>  ← fill in when migrating
#
# ---------------------------------------------------------------------------
# Expected interface (to be implemented)
# ---------------------------------------------------------------------------
#
#   from gin_bids_py_analysis.bids.file import BIDSFile
#
#   def load_ieeg(bids_file: BIDSFile) -> Any:
#       """Load iEEG data from a BIDSFile and return a data object."""
#       ...
#
#   class IEEGLoader:
#       """Stateful loader; accepts configuration (e.g. resample target, channel picks)."""
#       def load(self, bids_file: BIDSFile) -> Any:
#           ...
