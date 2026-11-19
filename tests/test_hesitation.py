"""
Tests for word retrieval latency and the filled/unfilled pause split.

These two measures exist to corroborate a high pause ratio, so the cases that
matter most are the ones that decide whether an escalation fires: a hesitation
before a content word, a filler bridging two silences, and the degenerate
inputs that must not produce a confident-looking number out of nothing.
"""

from __future__ import annotations

import pytest

from auracare.features.hesitation import (
    PauseBreakdown,
    is_content_word,
    is_filler,
    split_pauses,
    word_retrieval_latency,
)
from auracare.stt import Word


def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


# ---------------------------------------------------------------------------
# Lexical classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token", ["um", "Um", "uh", "erm", "hmm", "mm", "UH", "ah."]
)
def test_fillers_are_recognised(token):
    assert is_filler(token)


@pytest.mark.parametrize("token", ["hospital", "walking", "bright", "slowly"])
def test_content_words_are_recognised(token):
    assert is_content_word(token)


@pytest.mark.parametrize(
    "token", ["the", "of", "and", "is", "they", "um", "uh", "", "...", "123"]
)
def test_function_words_and_fillers_are_not_content(token):
    assert not is_content_word(token)


def test_content_word_ignores_case_and_punctuation():
    assert is_content_word("Hospital,")
    assert is_content_word("garden.")


# ---------------------------------------------------------------------------
# Word retrieval latency
# ---------------------------------------------------------------------------
def test_latency_measures_gap_before_content_words_only():
    # "the" at 1.0 follows a 0.8 s gap, but it is a function word and must not
    # count. "hospital" follows a 0.5 s gap and must.
    words = [
        w("I", 0.0, 0.2),
        w("the", 1.0, 1.2),        # 0.8 s gap, function word -> ignored
        w("hospital", 1.7, 2.2),   # 0.5 s gap, content word -> counted
    ]
    out = word_retrieval_latency(words)
    assert out.n_content_words == 1
    assert out.mean_latency_s == pytest.approx(0.5, abs=1e-6)
    assert out.max_latency_s == pytest.approx(0.5, abs=1e-6)


def test_fluent_speech_gives_near_zero_latency():
    words = [
        w("the", 0.0, 0.2),
        w("garden", 0.2, 0.6),
        w("looked", 0.6, 0.9),
        w("lovely", 0.9, 1.3),
    ]
    out = word_retrieval_latency(words)
    assert out.n_content_words == 3
    assert out.mean_latency_s == pytest.approx(0.0, abs=1e-6)
    assert out.n_hesitations == 0


def test_hesitations_are_counted_above_threshold():
    words = [
        w("the", 0.0, 0.2),
        w("garden", 0.9, 1.3),    # 0.7 s -> hesitation
        w("was", 1.3, 1.5),
        w("quiet", 2.1, 2.5),     # 0.6 s -> hesitation
        w("today", 2.5, 2.9),     # 0.0 s -> not
    ]
    out = word_retrieval_latency(words)
    assert out.n_hesitations == 2
    assert out.hesitation_rate == pytest.approx(2 / 3, abs=1e-6)


def test_negative_gap_from_alignment_overlap_is_clamped():
    """Aligners can overlap adjacent words; that is not a negative latency."""
    words = [w("the", 0.0, 0.5), w("garden", 0.4, 0.9)]
    out = word_retrieval_latency(words)
    assert out.mean_latency_s == pytest.approx(0.0, abs=1e-6)
    assert out.mean_latency_s >= 0.0


def test_filler_closes_the_gap_for_the_following_word():
    """
    The speaker was audibly occupied during "um", so the retrieval gap for
    "hospital" runs from the end of the filler, not from the previous word.
    """
    words = [
        w("the", 0.0, 0.2),
        w("um", 0.3, 0.9),
        w("hospital", 1.0, 1.5),
    ]
    out = word_retrieval_latency(words)
    # 1.0 - 0.9 = 0.1, not 1.0 - 0.2 = 0.8
    assert out.mean_latency_s == pytest.approx(0.1, abs=1e-6)


def test_no_words_reports_unavailable_rather_than_zero():
    out = word_retrieval_latency([])
    assert out.mean_latency_s is None
    assert out.notes and "unavailable" in out.notes[0]


def test_single_word_cannot_give_a_latency():
    out = word_retrieval_latency([w("hospital", 0.0, 0.5)])
    assert out.mean_latency_s is None


# ---------------------------------------------------------------------------
# Filled vs unfilled pauses
# ---------------------------------------------------------------------------
def test_silence_with_no_filler_is_unfilled():
    pauses = [(1.0, 3.0)]
    words = [w("garden", 0.0, 1.0), w("quiet", 3.0, 3.5)]
    out = split_pauses(pauses, words)
    assert out.n_unfilled == 1
    assert out.n_filled == 0
    assert out.mean_unfilled_s == pytest.approx(2.0, abs=1e-6)
    assert out.longest_unfilled_s == pytest.approx(2.0, abs=1e-6)


def test_filler_bridging_two_silences_merges_into_one_filled_pause():
    """
    Silence, "um", silence is one hesitation. Counting it as two short
    unfilled pauses would understate it and hide it from the 4 s rule.
    """
    pauses = [(1.0, 2.0), (2.5, 4.0)]
    words = [
        w("garden", 0.0, 1.0),
        w("um", 2.0, 2.5),
        w("quiet", 4.0, 4.5),
    ]
    out = split_pauses(pauses, words)
    assert out.n_filled == 1
    assert out.n_unfilled == 0
    assert out.filled_segments == [(1.0, 4.0)]


def test_mixed_pauses_are_split_correctly():
    pauses = [(1.0, 1.5), (3.0, 6.0)]
    words = [
        w("garden", 0.0, 1.0),
        w("um", 1.5, 1.8),      # fills the first pause
        w("was", 1.8, 2.0),
        w("quiet", 6.0, 6.5),   # second pause has no filler
    ]
    out = split_pauses(pauses, words)
    assert out.n_filled == 1
    assert out.n_unfilled == 1
    assert out.longest_unfilled_s == pytest.approx(3.0, abs=1e-6)


def test_long_unfilled_pause_is_reported_for_the_escalation_rule():
    """The 4 s single-silence trigger reads longest_unfilled_s."""
    pauses = [(1.0, 6.5)]
    words = [w("garden", 0.0, 1.0), w("quiet", 6.5, 7.0)]
    out = split_pauses(pauses, words)
    assert out.longest_unfilled_s == pytest.approx(5.5, abs=1e-6)
    assert out.longest_unfilled_s > 4.0


def test_no_pauses_gives_empty_breakdown():
    out = split_pauses([], [w("garden", 0.0, 1.0)])
    assert isinstance(out, PauseBreakdown)
    assert out.n_unfilled == 0
    assert out.mean_unfilled_s is None


def test_missing_word_timings_are_declared_not_assumed():
    """
    Without alignment, fillers cannot be found. The pauses are still reported
    as unfilled, but the caller is told why rather than being left to assume
    the recording simply had no fillers in it.
    """
    out = split_pauses([(1.0, 3.0)], [])
    assert out.n_unfilled == 1
    assert out.notes and "no word timings" in out.notes[0]


def test_zero_fillers_is_flagged_as_a_transcription_limit():
    """
    Whisper strips disfluencies, so a filled-pause count of zero says nothing
    about the speaker. Measured on a 10-minute spontaneous recording it
    returned 0 fillers across 606 words. Reporting "0 filled pauses" without
    that caveat would read as a clinical finding.
    """
    pauses = [(1.0, 2.0), (3.0, 4.0)]
    words = [w("garden", 0.0, 1.0), w("quiet", 2.0, 3.0)]
    out = split_pauses(pauses, words)
    assert out.n_filled == 0
    assert out.notes, "a zero filler count must carry its caveat"
    assert "limit of the transcription" in out.notes[0]
