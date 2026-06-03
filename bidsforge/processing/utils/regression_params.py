from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictorAffineTransform(BaseModel):
    """Affine transform applied to predictor values for one condition."""

    scale: float = Field(default=1.0)
    offset: float = Field(default=0.0)

    @field_validator("scale", "offset")
    @classmethod
    def _validate_finite(cls, value: float) -> float:
        if not float("-inf") < float(value) < float("inf"):
            raise ValueError("Predictor transform values must be finite.")
        return float(value)
