"""bidsforge: BIDS-based iEEG analysis pipeline."""

import logging

__version__ = "1.0.0"

# Prevent "No handlers could be found" warnings when the library is used
# without an application-level logging configuration (PEP 8 / Python docs).
logging.getLogger(__name__).addHandler(logging.NullHandler())
