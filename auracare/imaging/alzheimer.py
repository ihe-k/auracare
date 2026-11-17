"""
MRI dementia-stage classification.

The artifact ships from the project, under
`models/`, and the search order falls back to a triage checkout if one is
beside us, so the tab works out of the box and still picks up a newly trained
model if she retrains.

Deliberately free of Streamlit so the classification can be tested without a
browser.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import features as image_features
from .features import ImageDecodeError

# The artifact is a dict, not a bare estimator. These are the keys that
# matter; `train_model.py` in the triage project writes them.
REQUIRED_KEYS = ("classifier", "scaler")


@dataclass(frozen=True)
class Prediction:
    """One classification, or the reason there isn't one."""

    label: str = ""
    confidence: float = 0.0
    probabilities: List[Tuple[str, float]] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _read_pickle(path: Path) -> Any:
    """Load the artifact, gzipped or not, chosen by content not by name."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:  # type: ignore[operator]
        return pickle.load(fh)


def load_model(path: Path) -> Dict[str, Any]:
    """
    The classifier bundle from `path`.

    Raises ValueError rather than returning a half-model: a bundle missing
    its scaler would otherwise reach `classify` and fail there, one layer
    further from the cause.
    """
    model = _read_pickle(Path(path))
    if not isinstance(model, dict):
        raise ValueError(
            f"expected a dict of model parts, found {type(model).__name__}"
        )
    missing = [k for k in REQUIRED_KEYS if model.get(k) is None]
    if missing:
        raise ValueError(f"artifact is missing: {', '.join(missing)}")
    return model


def class_names(model: Dict[str, Any]) -> Optional[List[str]]:
    """
    The names the model was trained with, or None.

    Never guessed. A hard-coded list in the earlier version named a class the
    model cannot predict and omitted one it can, which mislabels every
    prediction while looking entirely healthy.
    """
    encoder = model.get("label_encoder")
    if encoder is not None and hasattr(encoder, "classes_"):
        return [str(c) for c in encoder.classes_]
    if model.get("classes"):
        return [str(c) for c in model["classes"]]
    return None


def classify(source: Any, model: Dict[str, Any]) -> Prediction:
    """
    Classify one MRI.

    `source` is anything the feature module can decode: a path, bytes, a file
    object, or a Streamlit upload.
    """
    classes = class_names(model)
    if classes is None:
        return Prediction(error="This model does not carry its class names.")

    classifier = model["classifier"]
    if not hasattr(classifier, "predict_proba"):
        # The version before last returned a flat 85% here, which reads on
        # screen as a measurement but was a constant.
        return Prediction(error="This model cannot report a confidence.")

    try:
        vector = image_features.features_for_image(source)
    except ImageDecodeError as exc:
        return Prediction(error=f"Could not read that image: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        return Prediction(error=f"Could not read that image: {exc}")

    scaler = model["scaler"]
    expected = getattr(scaler, "n_features_in_", vector.size)
    if vector.size != expected:
        return Prediction(
            error=(
                f"This image produced {vector.size:,} features but the model "
                f"expects {expected:,}. The artifact and the extraction code "
                "are out of step."
            )
        )

    probabilities = classifier.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
    if len(probabilities) != len(classes):
        return Prediction(
            error=(
                f"The model returned {len(probabilities)} scores for "
                f"{len(classes)} class names."
            )
        )

    order = np.argsort(probabilities)[::-1]
    ranked = [(classes[i], float(probabilities[i]) * 100.0) for i in order]
    best = ranked[0]
    return Prediction(label=best[0], confidence=best[1], probabilities=ranked)


def describe(model: Dict[str, Any]) -> Dict[str, Any]:
    """What the artifact says about itself, for the tab's caption."""
    scaler = model.get("scaler")
    return {
        "classes": class_names(model),
        "n_features": getattr(scaler, "n_features_in_", None),
        "train_accuracy": model.get("train_accuracy"),
        "test_accuracy": model.get("test_accuracy"),
        "confusion_matrix": model.get("confusion_matrix"),
    }


__all__ = [
    "Prediction",
    "REQUIRED_KEYS",
    "class_names",
    "classify",
    "describe",
    "load_model",
]
