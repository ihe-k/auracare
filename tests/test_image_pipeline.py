"""
The MRI classifier.
"""

import pathlib

import numpy as np
import pytest

from auracare import paths
from auracare.imaging import alzheimer
from auracare.imaging import features as image_features

MRI_DIR = pathlib.Path(__file__).parent / "data" / "mri"

# What she labelled each file. The filename is the only label available, and
# it is hers.
LABELLED = {
    "Non-demented.jpg": "Non Demented",
    "Very_mild_dementia.jpg": "Very mild Dementia",
    "Mild_dementia.jpg": "Mild Dementia",
    "Moderate_dementia.jpg": "Moderate Dementia",
}


@pytest.fixture(scope="module")
def model():
    artifact = paths.find_image_model()
    if artifact is None:
        pytest.skip("no classifier artifact present")
    return alzheimer.load_model(artifact)


# ---------------------------------------------------------------------------
# The copied feature module must not drift from the triage original
# ---------------------------------------------------------------------------
def test_the_copied_feature_module_matches_the_triage_original():
    """
    `imaging/features.py` is `image_features.py` from Cognitive_AI_Triage,
    copied so Auracare runs without that checkout beside it. A copy that
    drifts is worse than no copy: the two projects would score the same scan
    differently and nothing would look broken.
    """
    repo = paths.find_triage_repo()
    if repo is None:
        pytest.skip("no Cognitive_AI_Triage checkout to compare against")
    original = repo / "image_features.py"
    if not original.is_file():
        pytest.skip("triage checkout has no image_features.py")

    ours = (
        pathlib.Path(image_features.__file__)
        .read_text(encoding="utf-8")
        .split("# " + "-" * 75 + "\n", 2)[-1]
    )
    assert ours == original.read_text(encoding="utf-8"), (
        "imaging/features.py has drifted from the triage original; "
        "re-copy it rather than editing it here"
    )


# ---------------------------------------------------------------------------
# Finding the artifact
# ---------------------------------------------------------------------------
def test_the_artifact_ships_with_the_project():
    """
    
    """
    bundled = paths.PROJECT_ROOT / "models" / "alzheimer_classifier.pkl.gz"
    assert bundled.is_file(), "the classifier is not bundled under models/"


def test_the_bundled_artifact_is_what_is_found():
    found = paths.find_image_model()
    assert found is not None
    assert found.name in paths.IMAGE_MODEL_FILENAMES


def test_a_gzipped_artifact_is_recognised_by_content_not_by_name(tmp_path):
    """
    """
    import gzip
    import pickle

    payload = {"classifier": object(), "scaler": object()}
    misnamed = tmp_path / "alzheimer_classifier.pkl"
    with gzip.open(misnamed, "wb") as fh:
        pickle.dump(payload, fh)
    loaded = alzheimer.load_model(misnamed)
    assert set(loaded) == {"classifier", "scaler"}


def test_an_incomplete_artifact_is_rejected_where_it_is_loaded(tmp_path):
    """Not three layers later, inside a prediction."""
    import pickle

    path = tmp_path / "broken.pkl"
    path.write_bytes(pickle.dumps({"classifier": object()}))
    with pytest.raises(ValueError, match="scaler"):
        alzheimer.load_model(path)


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------
def test_class_names_come_from_the_artifact(model):
    names = alzheimer.class_names(model)
    assert names == [
        "Mild Dementia",
        "Moderate Dementia",
        "Non Demented",
        "PET",
        "Very mild Dementia",
    ]


def test_a_model_with_no_class_names_is_reported_not_guessed():
    """
    """
    result = alzheimer.classify(MRI_DIR / "Non-demented.jpg", {"classifier": 1, "scaler": 1})
    assert not result.ok
    assert "class names" in result.error


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", sorted(LABELLED))
def test_every_supplied_image_classifies_without_error(filename, model):
    result = alzheimer.classify(MRI_DIR / filename, model)
    assert result.ok, result.error
    assert result.label in alzheimer.class_names(model)
    assert 0.0 <= result.confidence <= 100.0
    assert len(result.probabilities) == 5
    assert result.probabilities[0][1] == pytest.approx(result.confidence)


def test_three_of_the_four_labelled_images_are_classified_correctly(model):
    """
    .
    """
    hits = {
        name: alzheimer.classify(MRI_DIR / name, model).label == want
        for name, want in LABELLED.items()
    }
    assert sum(hits.values()) == 3, hits
    assert hits["Moderate_dementia.jpg"] is False


def test_the_known_miss_keeps_the_right_answer_in_second_place(model):
    """If it ever falls further than second, the pipeline has changed."""
    result = alzheimer.classify(MRI_DIR / "Moderate_dementia.jpg", model)
    ranked = [name for name, _ in result.probabilities]
    assert ranked.index("Moderate Dementia") == 1


def test_the_four_images_do_not_all_get_the_same_answer(model):
    """
    """
    labels = {
        alzheimer.classify(MRI_DIR / name, model).label for name in LABELLED
    }
    assert len(labels) >= 3


def test_a_file_that_is_not_an_image_is_reported_not_crashed(model, tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not an MRI")
    result = alzheimer.classify(junk, model)
    assert not result.ok
    assert "Could not read" in result.error


def test_features_are_the_width_the_model_expects(model):
    vector = image_features.features_for_image(MRI_DIR / "Non-demented.jpg")
    assert vector.size == image_features.EXPECTED_N_FEATURES
    assert vector.size == model["scaler"].n_features_in_
    assert np.isfinite(vector).all()
