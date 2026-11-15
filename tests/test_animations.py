"""
Tests for the Annie animation clips.

The mapping of clip to task came from transcribing each file, not from the
filenames, which were sequential ids in an order that did not match the tasks.
These tests pin that mapping so a future re-export cannot silently attach the
wrong instruction to a task - a patient told to do the wrong manoeuvre would
invalidate the measurement without anything looking broken.
"""

from __future__ import annotations

from auracare import paths
from auracare.ui import TASK_SCRIPTS, script_for


def test_every_task_has_its_own_animation():
    for task_id, expected in (
        (1, "task1.mp4"),
        (2, "task2.mp4"),
        (3, "task3.mp4"),
        (4, "task4.mp4"),
    ):
        assert script_for(task_id).animation == expected


def test_no_two_tasks_share_a_clip():
    """
    Task 2 was first supplied as a byte-identical copy of the pa-ta-ka clip
    already in use for task 3, which would have told patients to repeat
    syllables during the forced-exhalation manoeuvre. The measurement would
    have been invalid with nothing visibly broken, so the distinctness of the
    files is pinned here, by content and not by name.
    """
    import hashlib

    digests = {}
    for script in TASK_SCRIPTS:
        if script.animation is None:
            continue
        path = paths.find_animation(script.animation)
        assert path is not None
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        assert digest not in digests, (
            f"task {script.task_id} ({script.animation}) is a duplicate of "
            f"task {digests[digest]}'s clip"
        )
        digests[digest] = script.task_id
    assert len(digests) == 4


def test_every_declared_animation_file_exists():
    for script in TASK_SCRIPTS:
        if script.animation is None:
            continue
        assert paths.find_animation(script.animation) is not None, (
            f"task {script.task_id} declares {script.animation}, which is missing"
        )


def test_intro_animation_is_present():
    assert paths.find_animation(paths.INTRO_ANIMATION) is not None


def test_missing_animation_resolves_to_none_rather_than_raising():
    """A missing clip degrades to the written prompt; it must not break."""
    assert paths.find_animation("does_not_exist.mp4") is None
    assert paths.find_animation(None) is None
    assert paths.find_animation("") is None
