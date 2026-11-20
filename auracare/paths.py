"""
Locating the project's own files and the sibling Cognitive_AI_Triage checkout.

Rather than hardcode a
path - the current `streamlit_inference.py` hardcodes `/tmp`, which cannot work
on Windows - this searches a short list of plausible locations and lets an
environment variable override everything.

    AURACARE_TRIAGE_REPO   path to the Cognitive_AI_Triage checkout
    AURACARE_MODEL_DIR     path to a directory holding the .pkl artifacts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"

SEVERITY_MODEL_FILENAMES = ("severity_model.pkl", "severity_model.pkl.2")
IMAGE_MODEL_FILENAMES = (
    "alzheimer_classifier.pkl.gz",
    "alzheimer_classifier.pkl",
    "alzheimer_classifier.pkl.2",
)


def _candidate_repo_roots() -> List[Path]:
    """Plausible locations for the Cognitive_AI_Triage checkout."""
    env = os.environ.get("AURACARE_TRIAGE_REPO")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        PROJECT_ROOT / "Cognitive_AI_Triage",
        PROJECT_ROOT.parent / "Cognitive_AI_Triage",
        PROJECT_ROOT.parent.parent / "Cognitive_AI_Triage",
        Path.cwd() / "Cognitive_AI_Triage",
        Path.cwd().parent / "Cognitive_AI_Triage",
    ]
    return candidates


def find_triage_repo() -> Optional[Path]:
    """The Cognitive_AI_Triage checkout, or None."""
    for path in _candidate_repo_roots():
        if (path / "streamlit_inference.py").exists() or (path / "artifacts").is_dir():
            return path.resolve()
    return None


def _model_search_dirs() -> List[Path]:
    dirs: List[Path] = []
    env = os.environ.get("AURACARE_MODEL_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    # Bundled first, so the prototype works unpacked on its own.
    dirs.append(PROJECT_ROOT / "models")
    repo = find_triage_repo()
    if repo:
        dirs += [
            repo / "artifacts",
            repo / "predictive_cognitive_final" / "artifacts",
        ]
    return dirs


def find_severity_model() -> Optional[Path]:
    """Path to the PHQ-8 severity-model artifact, or None."""
    for directory in _model_search_dirs():
        for name in SEVERITY_MODEL_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    return None


def find_image_model() -> Optional[Path]:
    """
    Path to the MRI classifier artifact, or None.

    Searched in the same order as the severity model, which puts the copy
    bundled under `models/` first. That ordering is the repair: the tab used
    to find nothing and tell the clinician to set AURACARE_TRIAGE_REPO, which
    is not a reasonable thing to ask of someone opening a prototype. A triage
    checkout beside us still wins over nothing, so a freshly trained model is
    picked up without moving files, and AURACARE_MODEL_DIR still overrides
    everything.
    """
    for directory in _model_search_dirs():
        for name in IMAGE_MODEL_FILENAMES:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()
    return None


def find_practice_files() -> Optional[Path]:
    """
    """
    repo = find_triage_repo()
    if repo and (repo / "Practice_Files").is_dir():
        return (repo / "Practice_Files").resolve()
    local = PROJECT_ROOT / "Practice_Files"
    return local.resolve() if local.is_dir() else None


def describe() -> dict:
    """Everything the UI needs to tell the user what it did and did not find."""
    repo = find_triage_repo()
    return {
        "project_root": str(PROJECT_ROOT),
        "triage_repo": str(repo) if repo else None,
        "severity_model": str(find_severity_model() or "") or None,
        "image_model": str(find_image_model() or "") or None,
        "practice_files": str(find_practice_files() or "") or None,
    }


__all__ = [
    "ASSETS_DIR",
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "describe",
    "find_image_model",
    "find_practice_files",
    "find_animation",
    "find_closing_crisis",
    "find_closing_emergency",
    "find_severity_model",
    "find_stimulus",
    "find_task_visual",
    "find_triage_repo",
]


STIMULUS_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_stimulus() -> Optional[Path]:
    """The Task 4 picture-description stimulus, whatever the client named it.

    """
    if not ASSETS_DIR.is_dir():
        return None
    candidates = [
        path
        for path in ASSETS_DIR.iterdir()
        if path.is_file()
        and path.stem.lower().startswith("stimulus")
        and path.suffix.lower() in STIMULUS_EXTENSIONS
    ]
    if not candidates:
        return None
    # Deterministic when several are present: the canonical name wins, so a
    # leftover file cannot silently displace the intended picture.
    canonical = [p for p in candidates if p.name == "stimulus_park.png"]
    return canonical[0] if canonical else sorted(candidates)[0]


ANIMATIONS_DIR = ASSETS_DIR / "animations"
INTRO_ANIMATION = "intro.mp4"

CLOSING_GOOD_ANIMATION = "Phq-8_compete_good.mp4"
CLOSING_REVIEW_ANIMATION = "Phq-8_complete_bad.mp4"
# Kept so an older assets folder still works. Deliberately not reached on the
# crisis and emergency endings: see `find_closing_crisis`.
CLOSING_ANIMATION = "closing.mp4"

# "Urgent" is two different endings here, and one clip cannot serve both.
#
#   PATHWAY_EMERGENCY is respiratory: strained breathing, triage now, 999.
#   PATHWAY_CRISIS    is a disclosure of self-harm: Samaritans, 116 123.
#
CLOSING_EMERGENCY_ANIMATIONS = (
    "Phq-8_complete_emergency.mp4",
    "Phq-8_compete_emergency.mp4",
    "closing_emergency.mp4",
)
# Still missing. The second of the two she said she was sending.
CLOSING_CRISIS_ANIMATIONS = (
    "Phq-8_complete_crisis.mp4",
    "Phq-8_compete_crisis.mp4",
    "closing_crisis.mp4",
)
PHQ8_INTRO_ANIMATION = "Phq-8_Introduction.mp4"


def find_animation(filename: Optional[str]) -> Optional[Path]:
    """An Annie animation clip in assets/animations/, or None.

    """
    if not filename:
        return None
    candidate = ANIMATIONS_DIR / filename
    return candidate.resolve() if candidate.is_file() else None


def _first_animation_present(filenames) -> Optional[Path]:
    for filename in filenames:
        found = find_animation(filename)
        if found is not None:
            return found
    return None


def find_closing_emergency() -> Optional[Path]:
    """Annie's closing clip for the respiratory emergency ending, or None.

    
    """
    return _first_animation_present(CLOSING_EMERGENCY_ANIMATIONS)


def find_closing_crisis() -> Optional[Path]:
    """Annie's closing clip for the mental health crisis ending, or None.

    """
    return _first_animation_present(CLOSING_CRISIS_ANIMATIONS)


TASK_VISUAL_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm",
)


def find_task_visual(task_id: int) -> Optional[Path]:
    """The clinician's own visual for one task, or None for the built-in cue.

    """
    if not ASSETS_DIR.is_dir():
        return None
    prefix = f"task{task_id}_visual"
    candidates = [
        path
        for path in ASSETS_DIR.iterdir()
        if path.is_file()
        and path.stem.lower().startswith(prefix)
        and path.suffix.lower() in TASK_VISUAL_EXTENSIONS
    ]
    if not candidates:
        return None
    # Deterministic when several are present, as with the stimulus: the exact
    # name wins, so a leftover copy cannot displace the intended file.
    exact = [p for p in candidates if p.stem.lower() == prefix]
    return exact[0] if exact else sorted(candidates)[0]
