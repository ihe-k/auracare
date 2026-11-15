"""
Robustness evidence for the corrected MRI feature pipeline.

`verify_image_fix.py` shows the fix on two ground-truth images. Two images is
thin evidence, and the client is being asked to trust a clinical-adjacent
classifier, so this widens the test the only honest way available without more
labelled data: it perturbs each image in ways that must not change the
diagnosis - small rotations, brightness and contrast shifts, mild crops and
sensor-grade noise - and checks the prediction survives.

A correctly standardised pipeline is stable under those perturbations. The
legacy pipeline is not: its features land tens of thousands of sigma from the
scaler's training distribution, so the classifier is extrapolating and the
label it returns is close to arbitrary.

Reported per pipeline:

  * accuracy against the filename ground truth, over all variants
  * label stability - how often the variants agree with the clean image
  * mean confidence, and whether it moves under perturbation
  * determinism - the identical input classified twice must agree exactly

Usage:
    python scripts/verify_image_robustness.py
    python scripts/verify_image_robustness.py --model path/to/artifact.pkl.gz
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auracare import paths  # noqa: E402
from auracare.imaging.alzheimer import (  # noqa: E402
    classify_image,
    extract_features_legacy_buggy,
    extract_training_features,
    load_image_model,
    preprocess_legacy_app,
    preprocess_like_training,
)

# Ground truth comes from the filename, which is how the practice files are
# labelled. Matching is loose because the model's class names vary between the
# 3-class release artifact and the 5-class in-repo one.
TRUTH_KEYWORDS = {
    "mild_dementia": "mild",
    "non-demented": "non",
}


def ground_truth(path: Path) -> str:
    stem = path.stem.lower()
    for key, truth in TRUTH_KEYWORDS.items():
        if key in stem:
            return truth
    return ""


def label_matches(label: str, truth: str) -> bool:
    """Loose match: the model's class names differ between artifacts."""
    low = label.lower()
    if truth == "non":
        # "Non Demented" / "Non-demented". Must not match "Very mild".
        return low.startswith("non")
    if truth == "mild":
        # "Mild Dementia" but NOT "Very mild Dementia" - those are distinct
        # stages, and conflating them would flatter the result.
        return low.startswith("mild")
    return False


# -- perturbations that must not change a diagnosis -------------------------


def _png_bytes(img: Image.Image) -> bytes:
    """decode_image() takes bytes, not a PIL object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def variants(path: Path) -> List[Tuple[str, bytes]]:
    """Clean image plus perturbations a real scanner/export could produce."""
    base = Image.open(path).convert("L")
    w, h = base.size
    out: List[Tuple[str, Image.Image]] = [("clean", base)]

    for angle in (-6, -3, 3, 6):
        out.append((f"rotate{angle:+d}", base.rotate(angle, resample=Image.BILINEAR)))

    for factor in (0.85, 0.92, 1.08, 1.15):
        arr = np.asarray(base, dtype=np.float32) * factor
        out.append(
            (f"bright x{factor:g}", Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
        )

    for factor in (0.85, 1.15):
        arr = np.asarray(base, dtype=np.float32)
        arr = (arr - arr.mean()) * factor + arr.mean()
        out.append(
            (f"contrast x{factor:g}", Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
        )

    for frac in (0.03, 0.06):
        dx, dy = int(w * frac), int(h * frac)
        out.append((f"crop {frac:.0%}", base.crop((dx, dy, w - dx, h - dy))))

    rng = np.random.default_rng(0)  # fixed seed: this script must be repeatable
    for sigma in (2.0, 5.0):
        arr = np.asarray(base, dtype=np.float32) + rng.normal(0, sigma, (h, w))
        out.append(
            (f"noise s={sigma:g}", Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
        )

    return [(name, _png_bytes(img)) for name, img in out]


def _class_names(bundle) -> List[str]:
    """Readable class names, aligned to the classifier's own output order.

    The in-repo artifact stores its classes label-encoded ('0'..'4'), so the
    encoder has to be inverted or every label reads as a bare digit. This
    mirrors what classify_image() does, so the two pipelines are compared on
    identical naming.
    """
    classes = [str(c) for c in bundle.classes]
    order = getattr(bundle.classifier, "classes_", None)
    if order is not None and len(order) == len(classes):
        try:
            classes = [str(n) for n in bundle.label_encoder.inverse_transform(order)]
        except Exception:
            pass
    return classes


def legacy_predict(img_source, bundle) -> Tuple[str, float]:
    """The pre-fix pipeline, through the same model and scaler."""
    legacy_img = preprocess_legacy_app(img_source)
    feats = extract_features_legacy_buggy(legacy_img, bundle.n_features)
    scaled = bundle.scaler.transform(
        np.asarray(feats, dtype=np.float32).reshape(1, -1)
    )
    probs = np.asarray(bundle.classifier.predict_proba(scaled)[0], dtype=np.float64)
    classes = _class_names(bundle)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def max_abs_z(img_source, bundle, legacy: bool) -> float:
    """How far the features sit from the scaler's training distribution."""
    if legacy:
        feats = extract_features_legacy_buggy(
            preprocess_legacy_app(img_source), bundle.n_features
        )
    else:
        feats = extract_training_features(preprocess_like_training(img_source))
    f = np.asarray(feats, dtype=np.float64).ravel()[: bundle.n_features]
    mean = np.asarray(bundle.scaler.mean_, dtype=np.float64)
    scale = np.asarray(bundle.scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    return float(np.abs((f - mean) / scale).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to a classifier artifact")
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else paths.find_image_model()
    if model_path is None:
        print("No classifier artifact found.")
        print("Set AURACARE_TRIAGE_REPO, or pass --model.")
        return 2

    practice = paths.find_practice_files()
    if practice is None:
        print("No Practice_Files directory found.")
        return 2

    images = sorted(
        p for p in practice.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and ground_truth(p)
    )
    if not images:
        print(f"No labelled MRI images in {practice}.")
        return 2

    bundle = load_image_model(model_path)
    classes = _class_names(bundle)

    print("=" * 78)
    print("MRI PIPELINE ROBUSTNESS")
    print("=" * 78)
    print(f"model   : {model_path}")
    print(f"classes : {len(classes)} -> {classes}")
    print(f"images  : {len(images)} labelled, {len(variants(images[0]))} variants each")
    print()

    totals: Dict[str, Dict[str, int]] = {
        "corrected": {"n": 0, "correct": 0, "stable": 0},
        "legacy": {"n": 0, "correct": 0, "stable": 0},
    }
    conf: Dict[str, List[float]] = {"corrected": [], "legacy": []}
    zmax: Dict[str, List[float]] = {"corrected": [], "legacy": []}
    predicted: Dict[str, set] = {"corrected": set(), "legacy": set()}

    for path in images:
        truth = ground_truth(path)
        vs = variants(path)

        clean_corrected = classify_image(vs[0][1], bundle, drop_non_diagnostic=False).label
        clean_legacy = legacy_predict(vs[0][1], bundle)[0]

        print("-" * 78)
        print(f"{path.name}   ground truth: {truth}")
        print("-" * 78)
        print(f"{'variant':<16} {'corrected':<26} {'legacy (pre-fix)':<26}")

        for name, img in vs:
            c = classify_image(img, bundle, drop_non_diagnostic=False)
            l_label, l_conf = legacy_predict(img, bundle)

            for key, label, confidence, clean in (
                ("corrected", c.label, c.confidence, clean_corrected),
                ("legacy", l_label, l_conf, clean_legacy),
            ):
                totals[key]["n"] += 1
                totals[key]["correct"] += int(label_matches(label, truth))
                totals[key]["stable"] += int(label == clean)
                conf[key].append(float(confidence))
                predicted[key].add(label)

            zmax["corrected"].append(max_abs_z(img, bundle, legacy=False))
            zmax["legacy"].append(max_abs_z(img, bundle, legacy=True))

            ok_c = "OK " if label_matches(c.label, truth) else "BAD"
            ok_l = "OK " if label_matches(l_label, truth) else "BAD"
            print(
                f"{name:<16} {ok_c} {c.label:<18} {c.confidence:4.2f}  "
                f"{ok_l} {l_label:<18} {l_conf:4.2f}"
            )
        print()

    # determinism: the same input twice must agree exactly
    det_img = variants(images[0])[0][1]
    a = classify_image(det_img, bundle, drop_non_diagnostic=False)
    b = classify_image(det_img, bundle, drop_non_diagnostic=False)
    deterministic = a.label == b.label and abs(a.confidence - b.confidence) < 1e-12

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'pipeline':<12} {'accuracy':>12} {'stability':>12} {'mean conf':>11} {'max |z|':>14}")
    for key in ("corrected", "legacy"):
        t = totals[key]
        acc = t["correct"] / t["n"] if t["n"] else 0.0
        stab = t["stable"] / t["n"] if t["n"] else 0.0
        print(
            f"{key:<12} {t['correct']:>4}/{t['n']:<3} {acc:>5.0%} "
            f"{stab:>11.0%} {np.mean(conf[key]):>11.2f} {max(zmax[key]):>14,.1f}"
        )
    print()
    print(f"distinct labels, corrected : {sorted(predicted['corrected'])}")
    print(f"distinct labels, legacy    : {sorted(predicted['legacy'])}")
    print(f"deterministic (same input twice) : {'yes' if deterministic else 'NO'}")
    print()

    # Two separate claims, which must not be conflated. The first is about the
    # fix and is provable here; the second is about clinical fitness and is
    # not, because there are only two labelled images.
    z_corrected = max(zmax["corrected"])
    z_legacy = max(zmax["legacy"])
    fix_holds = z_corrected < z_legacy / 10 and deterministic

    stab = totals["corrected"]["stable"] / max(totals["corrected"]["n"], 1)
    acc = totals["corrected"]["correct"] / max(totals["corrected"]["n"], 1)

    print("-" * 78)
    print("CLAIM 1 - the standardisation fix is real")
    print(
        f"  features moved from {z_legacy:,.0f} sigma off the training "
        f"distribution to {z_corrected:,.0f}."
    )
    print(f"  deterministic: {'yes' if deterministic else 'NO'}")
    print(f"  -> {'HOLDS' if fix_holds else 'DOES NOT HOLD'}")
    print()
    print("CLAIM 2 - the classifier is fit for clinical use")
    print(f"  accuracy {acc:.0%}, label stability under perturbation {stab:.0%}.")
    print(f"  labelled images available: {len(images)}.")
    print(
        "  -> NOT ESTABLISHED. Two images cannot support an accuracy claim, and"
    )
    print(
        "     the label changes under rotations and exposure shifts that carry"
    )
    print(
        "     no clinical meaning. Screening indication only, as the UI states."
    )
    print()
    print(
        "To move claim 2, the model needs a held-out labelled validation set,"
    )
    print("not more perturbations of the same two scans.")
    return 0 if fix_holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
