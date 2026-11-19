"""
Conversation phases, response-clip routing and the two input judgements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
# A task step is in exactly one of these at any moment. Without the state the
# same screen has to infer what it is doing from the presence or absence of a
# recording, which is what made the silence case and the repeat button
# impossible to add separately.
PHASE_PROMPT = "PROMPT"            # waiting for the patient to record
PHASE_AFFIRM = "AFFIRM"            # recording accepted, Annie acknowledges it
PHASE_SILENCE_RETRY = "SILENCE_RETRY"  # nothing audible arrived
PHASE_CLARIFY = "CLARIFY"          # the instruction is being given again

PHASES: Tuple[str, ...] = (
    PHASE_PROMPT,
    PHASE_AFFIRM,
    PHASE_SILENCE_RETRY,
    PHASE_CLARIFY,
)


# ---------------------------------------------------------------------------
# Response clips
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskMedia:
    """The clips Annie plays in response to the patient, per task.

    The prompt clip is deliberately absent: it already lives on
    `ui.TaskScript.animation`, and a second copy of that filename here would
    be a second place for the task-to-clip mapping to go wrong. That mapping
    was established by transcribing the files rather than trusting their
    names, and it is pinned by `tests/test_animations.py`.
    """

    affirm: Optional[str] = None
    clarify: Optional[str] = None


# The clinician supplied and confirmed the affirmation order on 16 Nov:
#   1 "That's perfect, thank you."
#   2 "Well done, take a quick moment to catch your breath."
#   3 "Wonderful, you're doing great."
#   4 "I really appreciate you walking me through that. Your check-in is
#      complete!"
# Clarification clips, supplied 17 Nov for all four tasks and transcribed
# before being wired, as with every clip here. Three matched their task and
# are installed:
#
#   1 "Take a deep breath and say ahh for as long as you comfortably can."
#   2 "Blow out as hard and fast as you can into the microphone, like blowing
#      out a birthday candle."
#   4 "Look at the picture on your screen and describe what you see happening
#      in your own words."
#
# Task 4's was byte-identical to the copy already installed on 16 Nov, which
# confirms that mapping a second time.
#
# Task 3's is deliberately NOT wired. It says "Follow the circle on your
# screen as it expands and shrinks in rhythm", which is a breathing and pacing
# instruction. Task 3 in this build is still pa-ta-ka, and the screen beside
# it reads "Match the rhythm: pa - ta - ka". Wiring it would have Annie give
# breathing instructions over a diadochokinetic task, which is the same class
# of fault as a mislabelled clip: the patient hears the wrong words and
# nothing looks broken.
#
# The clip is consistent with the Task 3 rework she has described (TASKS.md
# G2, pa-ta-ka removed in favour of a rest step), so it is most likely correct
# for where Task 3 is going and wrong for where it is now. It is held in the
# `today/` folder until she confirms. One line here wires it.
#
# Where a clarification clip is missing the CLARIFY phase falls back to
# re-playing the prompt clip and showing the written instruction, which is a
# weaker version of the same thing rather than a dead end.
TASK_MEDIA: Dict[int, TaskMedia] = {
    1: TaskMedia(affirm="task1_affirm.mp4", clarify="task1_clarify.mp3"),
    2: TaskMedia(affirm="task2_affirm.mp4", clarify="task2_clarify.mp3"),
    3: TaskMedia(affirm="task3_affirm.mp4"),
    4: TaskMedia(affirm="task4_affirm.mp4", clarify="task4_clarify.mp3"),
}

# What each affirmation clip says. Supplied by the clinician on 16 Nov, held
# here for two reasons: it is the written fallback shown if a clip will not
# play, and it is the record of which line belongs to which task, which the
# filenames alone do not establish.
#
# Task 4's line says the check-in is complete, but the PHQ-8 questions still
# follow it. That is her wording and it is left alone; it is raised with her
# rather than changed here.
AFFIRMATION_TEXT: Dict[int, str] = {
    1: "That's perfect, thank you.",
    2: "Well done, take a quick moment to catch your breath.",
    3: "Wonderful, you're doing great.",
    4: (
        "I really appreciate you walking me through that. Your check-in is "
        "complete!"
    ),
}

# Annie reading each PHQ-8 question, supplied 16 Nov. Keyed by the item key
# rather than by position, so reordering the instrument cannot silently attach
# the wrong recording to a question. Every file was transcribed before being
# wired up; the mapping is what the audio says, not what the filename claims.
#
#   phq_q1  little interest or pleasure in doing things
#   phq_q2  felt down, depressed or hopeless
#   phq_q3  trouble falling asleep, staying asleep, or sleeping too much
#   phq_q4  feeling tired or having little energy
#   phq_q5  poor appetite or overeating
#   phq_q6  feeling bad about yourself, a failure, let your family down
#   phq_q7  trouble concentrating, reading or watching television
#   phq_q8  moving or speaking slowly, or fidgety and restless
PHQ8_QUESTION_AUDIO: Dict[str, str] = {
    "PHQ8_NoInterest": "phq_q1.mp3",
    "PHQ8_Depressed": "phq_q2.mp3",
    "PHQ8_Sleep": "phq_q3.mp3",
    "PHQ8_Tired": "phq_q4.mp3",
    "PHQ8_Appetite": "phq_q5.mp3",
    "PHQ8_Failure": "phq_q6.mp3",
    "PHQ8_Concentrating": "phq_q7.mp3",
    "PHQ8_Moving": "phq_q8.mp3",
}


def question_audio(item_key: str):
    """The clip for one PHQ-8 item, or None to fall back to the browser voice."""
    return PHQ8_QUESTION_AUDIO.get(item_key)


# Shared across every task: there is one "it is quiet on my end" clip, not four.
QUIET_RETRY_AUDIO = "quiet_retry.mp3"


def media_for(task_id: int) -> TaskMedia:
    """The response clips for a task; an empty set rather than a KeyError."""
    return TASK_MEDIA.get(task_id, TaskMedia())


# ---------------------------------------------------------------------------
# Was that silence?
# ---------------------------------------------------------------------------
# A WebM or WAV container with no audio in it is still a few hundred bytes, so
# a byte floor catches the empty-file case before decoding is attempted.
MIN_RECORDING_BYTES = 1_200
MIN_RECORDING_SECONDS = 1.0
# Matches `check_quality`'s own silence threshold on purpose. Two different
# definitions of "near-silent" in one app would mean a recording could be
# rejected by one and accepted by the other.
SILENCE_FLOOR_DBFS = -55.0


def silence_reason(
    raw: bytes,
    audio=None,
    *,
    decoded: bool = True,
) -> Optional[str]:
    """
    Why this recording cannot be treated as an attempt, or None if it can.

    Covers the three ways nothing arrives: an empty file, a file too short to
    hold the manoeuvre, and a file of the right length with nothing audible in
    it, which is what a muted or unselected microphone produces. The third is
    the common one in practice and the only one the patient cannot see for
    themselves.

    `decoded=False` says the bytes would not decode at all, which is also
    treated as a silent attempt rather than an error: the patient needs to be
    asked to try again, not shown a stack trace.
    """
    if raw is not None and len(raw) < MIN_RECORDING_BYTES:
        return "the recording came through empty"
    if not decoded or audio is None:
        return "the recording could not be read"
    if audio.duration_s < MIN_RECORDING_SECONDS:
        return f"the recording was only {audio.duration_s:.1f} seconds long"
    if audio.rms_dbfs < SILENCE_FLOOR_DBFS:
        return (
            f"nothing audible was picked up ({audio.rms_dbfs:.0f} dBFS); the "
            "microphone may be muted or set to the wrong device"
        )
    return None


# ---------------------------------------------------------------------------
# Was that a request for help?
# ---------------------------------------------------------------------------
# Phrases that mean only one thing, whatever else is in the utterance.
HELP_PHRASES: Tuple[str, ...] = (
    "dont understand",
    "do not understand",
    "didnt understand",
    "not understanding",
    "dont know what",
    "do not know what",
    "dont get it",
    "say that again",
    "said that again",
    "repeat that",
    "repeat the question",
    "can you repeat",
    "could you repeat",
    "say it again",
    "what do i do",
    "what do you mean",
    "what was that",
    "im confused",
    "i am confused",
    "im lost",
    "no idea what",
)

# Single words that mean help only in a short utterance. "what", "again" and
# "how" are ordinary vocabulary in a picture description, so on their own they
# are evidence of nothing.
HELP_WORDS: Tuple[str, ...] = (
    "what",
    "how",
    "help",
    "pardon",
    "repeat",
    "again",
    "confused",
    "sorry",
    "huh",
    "eh",
)

# A plea is short. "Sorry, what do I do?" is six words; a thirty-second
# account of a park scene is sixty or more. The length test is what stops a
# normal Task 4 answer being thrown away, and the clinician's keyword list as
# written would have discarded most of them.
MAX_PLEA_WORDS = 6

_WORD_RE = re.compile(r"[a-z0-9]+")

# Apostrophes are deleted rather than treated as separators, so "don't"
# becomes "dont" and matches the phrase list. Splitting on them instead gave
# "don t understand", which no phrase matched: the plainest request for help
# there is was read as an attempt at the task. Straight and curly forms both,
# because speech-to-text output and typed text disagree about which to use.
_APOSTROPHES = str.maketrans(
    {"'": "", chr(0x2019): "", chr(0x02BC): "", "`": ""}
)


def normalise_utterance(text: Optional[str]) -> str:
    """Lowercase, apostrophes deleted, punctuation to spaces, space collapsed."""
    if not text:
        return ""
    return " ".join(_WORD_RE.findall(text.lower().translate(_APOSTROPHES)))


def looks_like_a_plea_for_help(text: Optional[str]) -> bool:
    """
    Did the patient ask for the instruction again instead of doing the task?

    Two tests, not one. An unambiguous phrase counts at any length; a single
    keyword counts only in a short utterance.

    The clinician's version matched keywords as substrings against the whole
    transcript, which fails twice over. "again" is inside "against", "how" is
    inside "shower" and "show", so ordinary words trip it; and with "what" and
    "how" on the list, a genuine thirty-second picture description would be
    read as confusion and discarded. Whole-word matching and the length test
    fix both.
    """
    words = normalise_utterance(text)
    if not words:
        return False

    for phrase in HELP_PHRASES:
        if phrase in words:
            return True

    tokens = words.split()
    if len(tokens) > MAX_PLEA_WORDS:
        return False
    return any(token in HELP_WORDS for token in tokens)


# ---------------------------------------------------------------------------
# How long to wait before asking again
# ---------------------------------------------------------------------------
# Measured from the supplied clips with PyAV. Annie is still speaking while
# her prompt clip plays, so a timer started when the screen renders fires
# mid-sentence: on Task 2 the clinician's seven-second window would have
# interrupted her eleven seconds before she finished the instruction. The wait
# is the clip plus the grace period.
#
# `tests/test_animations.py` pins each clip by content hash, so a re-export
# that changed a duration cannot pass unnoticed, and the durations themselves
# are checked against the files wherever a decoder is installed.
PROMPT_CLIP_SECONDS: Dict[int, float] = {
    1: 9.98,
    2: 18.30,
    3: 9.98,
    4: 9.66,
}

# The clinician's number, and a reasonable one: long enough to take a breath,
# short enough that the silence does not become uncomfortable.
SILENCE_GRACE_SECONDS = 7.0

# Used when a task has no measured clip duration, so a new task cannot
# accidentally get a zero-length window.
DEFAULT_PROMPT_SECONDS = 12.0


def quiet_window_seconds(task_id: int) -> float:
    """Seconds to wait, from the prompt appearing, before asking again."""
    clip = PROMPT_CLIP_SECONDS.get(task_id, DEFAULT_PROMPT_SECONDS)
    return float(clip) + SILENCE_GRACE_SECONDS


__all__ = [
    "AFFIRMATION_TEXT",
    "DEFAULT_PROMPT_SECONDS",
    "HELP_PHRASES",
    "HELP_WORDS",
    "MAX_PLEA_WORDS",
    "MIN_RECORDING_BYTES",
    "MIN_RECORDING_SECONDS",
    "PHQ8_QUESTION_AUDIO",
    "PHASES",
    "PROMPT_CLIP_SECONDS",
    "PHASE_AFFIRM",
    "PHASE_CLARIFY",
    "PHASE_PROMPT",
    "PHASE_SILENCE_RETRY",
    "QUIET_RETRY_AUDIO",
    "SILENCE_FLOOR_DBFS",
    "SILENCE_GRACE_SECONDS",
    "TASK_MEDIA",
    "TaskMedia",
    "looks_like_a_plea_for_help",
    "media_for",
    "normalise_utterance",
    "question_audio",
    "quiet_window_seconds",
    "silence_reason",
]
