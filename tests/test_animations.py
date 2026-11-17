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


# ---------------------------------------------------------------------------
# The response clips
# ---------------------------------------------------------------------------
def test_every_declared_response_clip_exists():
    """
    The affirmation and clarification clips go through the same lookup as the
    task animations. A filename in `flow.TASK_MEDIA` with no file behind it
    degrades silently to written text, which is the right fallback and the
    wrong thing to ship without noticing.
    """
    from auracare import flow

    for task_id, media in flow.TASK_MEDIA.items():
        for role, filename in (("affirm", media.affirm),
                               ("clarify", media.clarify)):
            if filename is None:
                continue
            assert paths.find_animation(filename) is not None, (
                f"task {task_id} declares a {role} clip {filename} that is "
                "not installed"
            )


def test_the_quiet_retry_clip_exists():
    from auracare import flow

    assert paths.find_animation(flow.QUIET_RETRY_AUDIO) is not None


def test_no_clip_is_used_for_two_different_things():
    """
    Extends the duplicate check to every clip the app can play. Task 2's
    prompt was first supplied as a byte-identical copy of task 3's, and an
    affirmation clip that is secretly a prompt clip would be the same class of
    fault: the patient hears the wrong words with nothing visibly broken.
    """
    import hashlib

    from auracare import flow

    declared = {}
    for script in TASK_SCRIPTS:
        if script.animation:
            declared[f"task {script.task_id} prompt"] = script.animation
    for task_id, media in flow.TASK_MEDIA.items():
        if media.affirm:
            declared[f"task {task_id} affirmation"] = media.affirm
        if media.clarify:
            declared[f"task {task_id} clarification"] = media.clarify
    declared["quiet retry"] = flow.QUIET_RETRY_AUDIO
    declared["intro"] = paths.INTRO_ANIMATION
    declared["phq-8 introduction"] = paths.PHQ8_INTRO_ANIMATION
    for key, filename in flow.PHQ8_QUESTION_AUDIO.items():
        declared[f"phq-8 {key}"] = filename

    seen = {}
    for role, filename in declared.items():
        path = paths.find_animation(filename)
        assert path is not None, f"{role}: {filename} is missing"
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        assert digest not in seen, f"{role} is a duplicate of {seen[digest]}"
        seen[digest] = role

    # 4 prompts + 4 affirmations + 1 clarification + quiet retry + intro
    # + the PHQ-8 introduction + its 8 questions.
    assert len(seen) == 20


# ---------------------------------------------------------------------------
# The PHQ-8 clips
# ---------------------------------------------------------------------------
def test_annie_reads_every_phq8_question():
    from auracare import flow

    assert len(flow.PHQ8_QUESTION_AUDIO) == 8
    for key, filename in flow.PHQ8_QUESTION_AUDIO.items():
        assert paths.find_animation(filename) is not None, (
            f"{key} declares {filename}, which is not installed"
        )


def test_the_question_clips_line_up_with_the_instrument():
    """
    Keyed by item key, not by position, so the two cannot drift apart. The
    mapping was established by transcribing all eight files; the filenames
    were not trusted, because an earlier supplied clip turned out to be a
    byte-identical copy of a different task's.
    """
    from auracare import flow
    from auracare.phq8 import PHQ8_ITEMS

    assert list(flow.PHQ8_QUESTION_AUDIO) == [item.key for item in PHQ8_ITEMS]


def test_an_item_with_no_clip_falls_back_rather_than_failing():
    from auracare import flow

    assert flow.question_audio("PHQ8_NotAnItem") is None


def test_the_phq8_introduction_clip_is_present():
    assert paths.find_animation(paths.PHQ8_INTRO_ANIMATION) is not None
