"""
Tests for the conversation phases, the quiet check and the help scan.

These are the three judgements that decide what Annie does next, and all three
are pure functions on purpose: a wrong answer here is a patient being told to
record again when they just finished, or a thirty-second answer being thrown
away, and neither should need a browser to catch.

Two of the tests are regressions against the clinician's own sketch rather
than against this code, and they are marked as such. Her keyword scan matched
substrings against the whole transcript, which both fires on ordinary words
and discards normal Task 4 answers.
"""

from __future__ import annotations

import inspect

import pytest

from auracare import flow
from auracare.audio_io import check_quality
from conftest import as_audio, concat, silence, vowel


# ---------------------------------------------------------------------------
# Phases and the clip map
# ---------------------------------------------------------------------------
def test_phases_are_distinct_and_cover_the_four_states():
    assert len(set(flow.PHASES)) == len(flow.PHASES) == 4
    assert set(flow.PHASES) == {
        flow.PHASE_PROMPT,
        flow.PHASE_AFFIRM,
        flow.PHASE_SILENCE_RETRY,
        flow.PHASE_CLARIFY,
    }


def test_every_task_has_an_affirmation_clip():
    for task_id in (1, 2, 3, 4):
        assert flow.media_for(task_id).affirm == f"task{task_id}_affirm.mp4"


def test_three_of_the_four_clarification_clips_are_wired():
    """
    She sent all four on 17 Nov. Three were transcribed and matched their
    task. Task 3's gives breathing instructions over a pa-ta-ka task, so it
    is held until she confirms the Task 3 rework, and that task still degrades
    to a replay of the prompt. This records the gap rather than hiding it.
    """
    assert flow.media_for(1).clarify == "task1_clarify.mp3"
    assert flow.media_for(2).clarify == "task2_clarify.mp3"
    assert flow.media_for(4).clarify == "task4_clarify.mp3"
    assert flow.media_for(3).clarify is None


def test_an_unknown_task_gets_an_empty_media_set_not_a_keyerror():
    media = flow.media_for(99)
    assert media.affirm is None and media.clarify is None


def test_affirmation_wording_is_recorded_for_all_four_tasks():
    assert sorted(flow.AFFIRMATION_TEXT) == [1, 2, 3, 4]
    assert all(text.strip() for text in flow.AFFIRMATION_TEXT.values())


# ---------------------------------------------------------------------------
# Was that silence?
# ---------------------------------------------------------------------------
def test_an_empty_file_is_silence():
    assert flow.silence_reason(b"") is not None
    assert flow.silence_reason(b"x" * 50) is not None


def test_bytes_that_will_not_decode_are_silence_rather_than_an_error():
    """
    A browser that hands back a container the decoder cannot read is the same
    problem as a muted microphone from the patient's point of view: nothing
    arrived. They need to be asked again, not shown an error.
    """
    reason = flow.silence_reason(b"x" * 5_000, None, decoded=False)
    assert reason is not None
    assert "could not be read" in reason


def test_a_recording_shorter_than_a_second_is_silence():
    audio = as_audio(vowel(120.0, 0.4))
    reason = flow.silence_reason(b"x" * 20_000, audio)
    assert reason is not None
    assert "0.4 seconds" in reason


def test_a_full_length_recording_with_nothing_audible_is_silence():
    """
    The common case in practice, and the only one the patient cannot see for
    themselves: the microphone is muted or pointed at the wrong device, so the
    file is the right length and completely empty.
    """
    audio = as_audio(silence(8.0))
    reason = flow.silence_reason(b"x" * 100_000, audio)
    assert reason is not None
    assert "nothing audible" in reason


def test_a_usable_recording_has_no_silence_reason():
    audio = as_audio(concat(silence(0.3), vowel(120.0, 9.0), silence(0.3)))
    assert flow.silence_reason(b"x" * 300_000, audio) is None


def test_the_silence_floor_matches_the_one_quality_checking_uses():
    """
    Two different definitions of near-silent would mean a recording could be
    waved through by one check and rejected by the other.
    """
    default = inspect.signature(check_quality).parameters["silence_dbfs"].default
    assert flow.SILENCE_FLOOR_DBFS == default


# ---------------------------------------------------------------------------
# Was that a request for help?
# ---------------------------------------------------------------------------
PLEAS = [
    "What?",
    "Sorry?",
    "Pardon?",
    "Sorry, what do I do?",
    "I don't understand.",
    "I dont understand what you want",
    "Can you repeat that please",
    "Could you repeat the question",
    "Could you say that again",
    "What do you mean?",
    "I'm confused",
    "Help",
    "Huh?",
    "I have no idea what you are asking me to do here, sorry",
]


@pytest.mark.parametrize("text", PLEAS)
def test_a_request_for_help_is_recognised(text):
    assert flow.looks_like_a_plea_for_help(text) is True


NORMAL_ANSWERS = [
    # The regression that matters. Every one of these contains a word from the
    # clinician's keyword list, and every one is a perfectly good answer.
    "I can see a park with two children playing on the swings and a dog "
    "running about near the fence. The mother looks happy and relaxed.",
    "There is a boy leaning against the railings watching the ducks, and I "
    "think he feels left out somehow.",
    "A woman is coming out of the shower block with a towel, and I am not "
    "sure how old the children are, maybe seven or eight.",
    "The weather looks lovely however there are clouds in the distance.",
    "Ahhhhh",
    "pa ta ka pa ta ka pa ta ka pa ta ka",
    "The family are having a picnic and everyone seems to be enjoying it.",
]


@pytest.mark.parametrize("text", NORMAL_ANSWERS)
def test_a_normal_answer_is_not_a_request_for_help(text):
    assert flow.looks_like_a_plea_for_help(text) is False


def test_substring_matches_do_not_count():
    """
    Regression against the clinician's version, which tested `kw in text`.
    "again" sits inside "against", "how" inside "shower" and "however", and
    "what" inside nothing useful, so ordinary vocabulary tripped the scan.
    """
    for text in ("leaning against the wall", "the shower block", "however"):
        assert flow.looks_like_a_plea_for_help(text) is False


def test_a_long_answer_is_never_read_as_confusion():
    """
    Regression against the clinician's version, which had both "what" and
    "how" on the keyword list with no length test. A thirty-second picture
    description almost always contains one of them, so most valid Task 4
    recordings would have been discarded before they were measured.
    """
    long_answer = " ".join(["what", "how"] + ["word"] * 60)
    assert flow.looks_like_a_plea_for_help(long_answer) is False


def test_an_unambiguous_phrase_counts_at_any_length():
    """Length only guards the single keywords; a phrase means what it says."""
    text = " ".join(["word"] * 40) + " sorry I don't understand what you want"
    assert flow.looks_like_a_plea_for_help(text) is True


def test_nothing_said_is_not_a_request_for_help():
    for text in (None, "", "   ", "..."):
        assert flow.looks_like_a_plea_for_help(text) is False


def test_punctuation_and_apostrophes_do_not_change_the_answer():
    assert flow.normalise_utterance("I don't, understand!") == "i dont understand"
    # Deleting the apostrophe rather than splitting on it is what makes the
    # phrase list match; splitting gave "don t understand", which matched nothing.
    assert flow.looks_like_a_plea_for_help("I DON'T UNDERSTAND!!") is True


# ---------------------------------------------------------------------------
# How long to wait
# ---------------------------------------------------------------------------
def test_the_quiet_window_always_clears_the_prompt_clip():
    """
    A timer started when the screen renders fires while Annie is still
    talking. On Task 2 the clinician's seven-second window would have
    interrupted her eleven seconds before she finished the instruction.
    """
    for task_id in (1, 2, 3, 4):
        window = flow.quiet_window_seconds(task_id)
        clip = flow.PROMPT_CLIP_SECONDS[task_id]
        assert window == pytest.approx(clip + flow.SILENCE_GRACE_SECONDS)
        assert window > clip


def test_an_unlisted_task_still_gets_a_sane_window():
    assert flow.quiet_window_seconds(99) > flow.SILENCE_GRACE_SECONDS


def test_declared_clip_durations_match_the_installed_files():
    """
    The durations are measured values held in code. If a clip is re-exported
    at a different length the window silently becomes wrong, so it is checked
    against the files wherever a container decoder is installed.
    """
    av = pytest.importorskip("av", reason="PyAV not installed")
    from auracare import paths
    from auracare.ui import script_for

    for task_id, declared in flow.PROMPT_CLIP_SECONDS.items():
        path = paths.find_animation(script_for(task_id).animation)
        assert path is not None
        with av.open(str(path)) as container:
            actual = container.duration / av.time_base
        assert actual == pytest.approx(declared, abs=0.1), (
            f"task {task_id}: declared {declared}s, file is {actual:.2f}s"
        )
