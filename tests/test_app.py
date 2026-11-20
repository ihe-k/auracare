"""
Streamlit app smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auracare.features.ddk import analyse_ddk
from auracare.features.expiratory import analyse_expiratory
from auracare.features.fluency import analyse_fluency
from auracare.features.phonation import analyse_phonation
from auracare.phq8 import ITEM_KEYS
from auracare.stt import Transcript
from conftest import (
    am_syllable_train,
    as_audio,
    concat,
    noise_burst,
    silence,
    speech_with_pauses,
    vowel,
)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 120


def run_app(**state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    
    state.setdefault("welcome_done", True)
    
    state.setdefault("opening_done", {1, 2, 3, 4})
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def healthy_results(transcript_text: str | None = None):
    t4_audio = as_audio(
        speech_with_pauses([5.0, 5.0], [1.5], lead_s=0.5, trail_s=0.5)
    )
    transcript = (
        Transcript(text=transcript_text, available=True, backend="manual")
        if transcript_text
        else None
    )
    return {
        1: analyse_phonation(
            as_audio(concat(silence(0.3), vowel(120.0, 9.0), silence(0.3)))
        ),
        2: analyse_expiratory(
            as_audio(concat(silence(0.3), noise_burst(1.5), silence(0.3)))
        ),
        3: analyse_ddk(as_audio(am_syllable_train(6.0, 4.0))),
        4: analyse_fluency(t4_audio, transcript=transcript),
    }


def emergency_results():
    results = healthy_results()
    results[1] = analyse_phonation(
        as_audio(concat(silence(0.3), vowel(135.0, 1.2), silence(0.3)))
    )
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_app_starts_without_exception():
    at = run_app()
    assert not at.exception
    assert at.title[0].value.startswith("\U0001fa7a Auracare")


def test_all_tabs_render():
    at = run_app()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    
    assert labels == [
        "Check-in",
        "Results",
        "Clinical",
        "MRI",
        "Method",
    ]


def test_tab_labels_stay_short_enough_to_fit():
    """Guards the fix above: long labels push 'Check-in' off screen."""
    at = run_app()
    assert not at.exception
    for tab in at.tabs:
        assert len(tab.label) <= 10, (
            f"tab label {tab.label!r} is long enough to risk overflowing the "
            "tab strip on a narrow window"
        )


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_each_task_step_renders(step):
    at = run_app(step=step)
    assert not at.exception, f"step {step} raised: {at.exception}"


def test_phq8_step_renders():
    """The questions now come one per screen, after an introduction."""
    at = run_app(step=5)
    assert not at.exception


def test_completion_step_renders():
    at = run_app(step=6, finished=True, results=healthy_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception


def test_patient_and_clinical_views_render_with_results():
    at = run_app(
        step=6,
        finished=True,
        results=healthy_results(),
        phq8_responses={k: 1 for k in ITEM_KEYS},
    )
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Auracare" in body


def test_views_render_before_any_recording():
    """Empty state must not raise; it should invite the user to start."""
    at = run_app()
    assert not at.exception
    info = " ".join(i.value for i in at.info)
    assert "Complete the check-in first" in info



EMERGENCY_BANNER = "based on your check-in, your breathing sounds strained"
EMERGENCY_ROUTING = "routing your assessment to emergency triage"


def _error_text(at) -> str:
    return " ".join(e.value for e in at.error).lower()


def test_no_emergency_banner_before_anything_is_measured():
    """
    
    """
    at = run_app()
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER not in text
    assert EMERGENCY_ROUTING not in text


def test_no_emergency_banner_when_recordings_are_unmeasurable():
    """Silent recordings must not be read as a breach."""
    quiet = as_audio(silence(10.0))
    results = {
        1: analyse_phonation(quiet),
        2: analyse_expiratory(quiet),
        3: analyse_ddk(quiet),
        4: analyse_fluency(quiet),
    }
    at = run_app(step=6, finished=True, results=results,
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER not in text
    assert EMERGENCY_ROUTING not in text


def test_emergency_banner_appears_when_threshold_is_breached():
    at = run_app(step=6, finished=True, results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = _error_text(at)
    assert EMERGENCY_BANNER in text
    assert EMERGENCY_ROUTING in text


def test_completion_message_suppressed_during_escalation():
    """
    """
    at = run_app(step=6, finished=True, results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "Check-in complete" not in success_text


def test_completion_message_shown_on_a_routine_result():
    at = run_app(step=6, finished=True, results=healthy_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    success_text = " ".join(s.value for s in at.success)
    assert "Check-in complete" in success_text


def test_crisis_language_surfaces_support_resources():
    results = healthy_results(
        transcript_text="honestly I think I would be better off dead"
    )
    at = run_app(step=6, finished=True, results=results,
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    text = " ".join(e.value for e in at.error)
    assert "116 123" in text or "harming yourself" in text


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def test_threshold_controls_are_present_and_editable():
    at = run_app()
    assert not at.exception
    keys = {n.key for n in at.number_input}
    assert "thr_task1_phonation_critical_s" in keys
    assert "thr_task4_pause_ratio_critical" in keys


def test_changing_a_threshold_changes_the_outcome():
    """A 4 s phonation is fine by default and a breach at a 5 s threshold."""
    results = healthy_results()
    results[1] = analyse_phonation(
        as_audio(concat(silence(0.3), vowel(130.0, 4.0), silence(0.3)))
    )
    base = run_app(step=6, finished=True, results=results,
                   phq8_responses={k: 0 for k in ITEM_KEYS})
    assert "emergency triage" not in " ".join(e.value for e in base.error).lower()

    strict = run_app(step=6, finished=True, results=results,
                     phq8_responses={k: 0 for k in ITEM_KEYS})
    strict.number_input(key="thr_task1_phonation_critical_s").set_value(5.0).run()
    assert not strict.exception
    assert "emergency triage" in " ".join(e.value for e in strict.error).lower()


def test_patient_id_is_configurable():
    at = run_app()
    assert not at.exception
    assert at.text_input(key="patient_id").value == "NHS-948-204-11"


def test_task_can_be_skipped_without_a_recording():
    """
    """
    at = run_app()
    assert not at.exception

    labels = [b.label for b in at.button]
    assert any("Skip Task" in label for label in labels), (
        f"no skip control on an un-recorded task; buttons were {labels}"
    )


def test_skipping_a_task_advances_the_flow():
    at = run_app()
    assert not at.exception
    start = at.session_state["step"]

    skip = next(b for b in at.button if "Skip Task" in b.label)
    skip.click().run()

    assert at.session_state["step"] == start + 1
    assert start in at.session_state["skipped"]


# ---------------------------------------------------------------------------
# Conversation phases
# ---------------------------------------------------------------------------
# `st.audio_input` has no scripted equivalent, but its session-state key does:
# assigning a file-like object under the widget's key is what the widget would
# have returned, so the whole judge-and-route path runs for real.
class _Recording:
    """Stands in for the UploadedFile st.audio_input hands back."""

    def __init__(self, data: bytes, name: str = "rec.wav"):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def wav_bytes(samples, sample_rate: int = 16_000) -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(
        buffer, np.asarray(samples, dtype="float32"), sample_rate,
        format="WAV", subtype="PCM_16",
    )
    return buffer.getvalue()


def good_phonation() -> bytes:
    return wav_bytes(concat(silence(0.3), vowel(120.0, 9.0), silence(0.3)))


def test_a_task_starts_in_the_prompt_phase():
    at = run_app()
    assert at.session_state["phase"] == "PROMPT"
    assert at.session_state["phase_step"] == 1


def test_a_phase_does_not_follow_the_patient_into_the_next_task():
    """
    
    """
    at = run_app(step=2, phase="CLARIFY", phase_step=1)
    assert at.session_state["phase"] == "PROMPT"
    assert at.session_state["phase_step"] == 2


def test_the_repeat_button_replays_the_instruction():
    at = run_app()
    at = at.button(key="repeat_1").click().run()
    assert at.session_state["phase"] == "CLARIFY"
    assert any("Task 1 again" in info.value for info in at.info)


def test_carrying_on_from_a_repeat_returns_to_the_prompt():
    at = run_app(phase="CLARIFY", phase_step=1)
    at = at.button(key="clarify_done_1").click().run()
    assert at.session_state["phase"] == "PROMPT"


def test_task_four_repeats_with_its_own_clarification_clip():
    """Task 4 has a clip; tasks 1 to 3 fall back to replaying the prompt."""
    at = run_app(step=4, phase="CLARIFY", phase_step=4)
    assert len(at.get("audio")) >= 1


def test_a_silent_recording_is_not_measured_and_asks_for_another():
    """
    The muted-microphone case: a full-length file with nothing in it. The
    patient cannot tell, so Annie has to.
    """
    at = run_app(**{"rec_1": _Recording(wav_bytes(silence(8.0)))})
    assert at.session_state["phase"] == "SILENCE_RETRY"
    assert 1 not in at.session_state["results"]
    reason = at.session_state["rejected"][1][1]
    assert "nothing audible" in reason


def test_a_recording_too_short_to_hold_the_task_asks_for_another():
    at = run_app(**{"rec_1": _Recording(wav_bytes(vowel(120.0, 0.4)))})
    assert at.session_state["phase"] == "SILENCE_RETRY"
    assert 1 not in at.session_state["results"]


def test_bytes_that_will_not_decode_ask_for_another_rather_than_erroring():
    at = run_app(**{"rec_1": _Recording(b"not audio at all" * 500)})
    assert at.session_state["phase"] == "SILENCE_RETRY"
    assert not at.exception


def test_the_quiet_prompt_offers_both_a_retry_and_an_explanation():
    at = run_app(**{"rec_1": _Recording(wav_bytes(silence(8.0)))})
    keys = {button.key for button in at.button}
    assert {"quiet_ready_1", "quiet_explain_1"} <= keys


def test_the_same_rejected_recording_is_only_judged_once():
    """
    `st.audio_input` hands the same file back on every rerun. Without the
    fingerprint guard, setting a phase and rerunning sets it again
    immediately, and the app never settles.
    """
    at = run_app(**{"rec_1": _Recording(wav_bytes(silence(8.0)))})
    first = at.session_state["rejected"][1]
    at = at.button(key="quiet_ready_1").click().run()
    assert at.session_state["phase"] == "PROMPT"
    assert at.session_state["rejected"][1] == first
    assert not at.exception


def test_a_usable_recording_is_acknowledged_before_the_numbers():
    at = run_app(**{"rec_1": _Recording(good_phonation())})
    assert at.session_state["phase"] == "AFFIRM"
    assert 1 in at.session_state["results"]
    # The affirmation clip, and the continue button underneath it.
    assert len(at.get("video")) >= 1
    assert any(
        "Continue to Task 2" in button.label for button in at.button
    )


def test_a_usable_recording_clears_an_earlier_rejection():
    at = run_app(
        rejected={1: ("stale-fingerprint", "nothing audible was picked up")},
        **{"rec_1": _Recording(good_phonation())},
    )
    assert at.session_state["phase"] == "AFFIRM"
    assert 1 not in at.session_state["rejected"]


def test_finishing_a_task_stops_its_quiet_wait():
    """
    The quiet wait belongs to the prompt screen. Left running once the task
    moved to the affirmation, it stayed armed with a deadline that had long
    since passed, and the watchdog fragment's last tick spent it on whatever
    screen was showing by then.
    """
    at = run_app(**{"rec_1": _Recording(good_phonation())})
    assert at.session_state["phase"] == "AFFIRM"
    assert 1 not in at.session_state["silence_deadline"]
    assert not at.exception


def test_leaving_a_task_takes_its_quiet_wait_with_it():
    at = run_app(**{"rec_1": _Recording(good_phonation())})
    at = at.button(key="affirm_next_1").click().run()
    assert at.session_state["step"] == 2
    assert 1 not in at.session_state["silence_deadline"]
    assert at.session_state["phase"] == "PROMPT"
    assert not at.exception


def test_the_quiet_watchdog_checks_it_is_still_the_screen_it_started_on():
    """
    A fragment's timer lives in the browser, so one more tick can arrive
    after the task it was armed for has been left behind. Carrying the
    finished task's step, it finds no recorder - Streamlit has dropped the
    widget - reads the absence as silence, and fires the quiet prompt over
    the top of the next task while Annie is still introducing it. That is the
    19 Nov report, and these are the guards against it.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def quiet_watchdog("):]
    body = body[: body.index("def arm_quiet_watchdog(")]
    for guard in (
        'st.session_state.get("step", 0)',
        'st.session_state.get("phase_step") != step',
        'st.session_state.get("phase") != flow.PHASE_PROMPT',
        'st.session_state.get("silence_deadline", {}).get(step) != deadline',
        'st.session_state.get("opening_done", set())',
    ):
        assert guard in body, guard
    # And they all come before the one thing that changes the screen.
    assert body.index('st.session_state.get("opening_done", set())') < body.index(
        "set_phase(step, flow.PHASE_SILENCE_RETRY)"
    )


def test_the_quiet_wait_does_not_start_until_annie_has_spoken():
    """
    Her report of 19 and 20 Nov: "it is quiet on my end" arriving as Annie is
    about to talk. Reproduced on 20 Nov by refusing `play()` the way Safari
    does - her clip sits at 00:00 behind a play control while the clock runs
    out behind it, because the clock used to start when the recorder
    appeared rather than when the instruction was actually given.

    On a browser that starts the clips those are the same moment, which is
    why every earlier check missed it.
    """
    at = run_app()
    assert at.session_state["phase"] == "PROMPT"
    assert not at.session_state["silence_deadline"], (
        "the clock must not be running before Annie has been heard"
    )
    assert not at.exception


def test_the_quiet_wait_starts_once_annie_has_spoken():
    """And it does still start, which is the whole point of the feature."""
    at = run_app(annie_spoke={1})
    assert 1 in at.session_state["silence_deadline"]
    assert not at.exception


def test_the_quiet_wait_is_still_switchable_off_after_annie_speaks():
    at = run_app(annie_spoke={1}, watch_for_silence=False)
    at.run()
    assert not at.session_state["silence_deadline"]
    assert not at.exception


def test_a_task_with_no_clip_has_nothing_to_wait_for():
    """
    Not every task is guaranteed a clip. One without must not disable the
    quiet prompt for itself, or a missing file would silently turn the
    feature off for that task alone.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def annie_has_spoken("):]
    body = body[: body.index("def arm_quiet_watchdog(")]
    gate = body.index("if clip is None:")
    assert "st.session_state.annie_spoke.add(step)" in body[gate:gate + 200]


def test_the_control_that_reports_annie_is_a_real_button():
    """
    Every automatic step in this app is also a button, so a script that never
    runs cannot strand anyone. If this one is never pressed the recorder, the
    written instruction and "Say that again" are all still there, and the
    only thing lost is a prompt that would have been wrong.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    body = source[source.index("def annie_has_spoken("):]
    body = body[: body.index("def arm_quiet_watchdog(")]
    assert "st.container(key=ui.SPOKE_CONTAINER_KEY)" in body
    assert 'st.button("Annie has started"' in body
    assert "ui.press_when_clip_plays(" in body


def test_starting_over_forgets_that_annie_spoke():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    reset = source[source.index("def reset_all()"):]
    reset = reset[: reset.index("init_state()")]
    assert '"annie_spoke"' in reset


def test_the_quiet_watchdog_can_be_switched_off():
    at = run_app(watch_for_silence=False)
    at.run()
    assert at.session_state["phase"] == "PROMPT"
    assert not at.session_state["silence_deadline"]
    assert not at.exception


def test_only_the_current_prompt_is_shown_on_the_check_in_screen():
    """
    
    """
    import html

    from auracare.ui import script_for

    at = run_app(step=3, results=healthy_results())
    # The class name also appears in the stylesheet, so match the element.
    bubbles = [
        m.value for m in at.markdown
        if 'class="aura-agent"' in m.value
    ]
    assert len(bubbles) == 1
    # The bubble escapes its text, so the comparison has to as well.
    assert html.escape(script_for(3).prompt) in bubbles[0]
    assert html.escape(script_for(1).prompt) not in bubbles[0]


def test_the_whole_exchange_is_still_readable_in_the_clinical_view():
    """
    The log is no longer rendered on the check-in screen, so it needs a reader
    somewhere or the telemetry lines are written and never seen.
    """
    at = run_app(
        step=5,
        results=healthy_results(),
        log=[("agent", "hold the ah"), ("system", "telemetry - phonation 9.0 s")],
    )
    labels = [e.label for e in at.get("expander")]
    assert any("Session log" in label for label in labels)


# ---------------------------------------------------------------------------
# The closing screen
# ---------------------------------------------------------------------------
# The pathway sets, so the four branches are driven by what actually produces
# them rather than by setting the pathway directly.
ROUTINE = dict(results=healthy_results(), phq8_responses={k: 0 for k in ITEM_KEYS})
REVIEW = dict(results=healthy_results(), phq8_responses={k: 2 for k in ITEM_KEYS})


def closing(**state):
    return run_app(step=6, finished=True, **state)


def test_the_routine_closing_says_where_the_results_went():
    at = closing(**ROUTINE)
    text = " ".join(s.value for s in at.success)
    assert "added to your health record" in text


def test_the_closing_screen_does_not_tell_the_patient_they_are_well():
    """
    
    """
    at = closing(**ROUTINE)
    text = " ".join(s.value for s in at.success).lower()
    for claim in ("mainly fine", "everything is fine", "you are well",
                  "nothing to worry"):
        assert claim not in text


def test_her_opening_line_is_on_the_routine_closing():
    at = closing(**ROUTINE)
    from auracare import ui

    assert any(ui.agent_bubble(ui.SCREENING_THANKS) in m.value
               for m in at.markdown)


def test_her_opening_line_is_formatted_like_the_tasks():
    """
    
    """
    at = closing(**ROUTINE)
    from auracare import ui

    bubble = ui.agent_bubble(ui.SCREENING_THANKS)
    assert any(bubble in m.value for m in at.markdown)
    assert not any(
        m.value.strip().startswith("#### ") and ui.SCREENING_THANKS in m.value
        for m in at.markdown
    )


def test_her_opening_line_is_also_on_a_clinician_review():
    """A review is not an emergency, and the warm line reads fine before it."""
    at = closing(**REVIEW)
    from auracare import ui

    assert any(ui.agent_bubble(ui.SCREENING_THANKS) in m.value
               for m in at.markdown)
    assert any(
        "clinician to look at your results" in w.value for w in at.warning
    )


def test_her_opening_line_is_not_spoken_over_an_emergency():
    """
    
    """
    from auracare import ui

    at = closing(results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not any(ui.SCREENING_THANKS in m.value for m in at.markdown)
    assert any("emergency triage" in e.value for e in at.error)


def test_her_opening_line_is_not_shown_over_a_crisis_message():
    from auracare import ui

    at = closing(
        results=healthy_results(
            transcript_text="honestly I think I would be better off dead"
        ),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )
    assert not any(ui.SCREENING_THANKS in m.value for m in at.markdown)
    text = " ".join(e.value for e in at.error)
    assert "116 123" in text or "harming yourself" in text


def test_the_emergency_closing_renders_at_all():
    """
    
    """
    at = closing(results=emergency_results(),
                 phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception


def test_the_closing_screen_works_whether_or_not_the_clip_has_arrived():
    """
    
    """
    from auracare import paths

    at = closing(**ROUTINE)
    assert not at.exception
    expected = 1 if paths.find_animation(paths.CLOSING_GOOD_ANIMATION) else 0
    assert len(at.get("video")) == expected


def test_the_closing_caption_points_at_tabs_that_exist():
    """
    
    """
    import re

    at = closing(**ROUTINE)
    labels = {tab.label for tab in at.get("tab")}
    captions = [c.value for c in at.caption if "tabs above" in c.value]
    assert len(captions) == 1
    named = re.findall(r"\*\*(.+?)\*\*", captions[0])
    assert named
    for name in named:
        assert name in labels, f"caption names a tab {name!r} that does not exist"


# ---------------------------------------------------------------------------
# The PHQ-8, one question per screen
# ---------------------------------------------------------------------------
def questions(**state):
    """The questionnaire with the introduction already dismissed."""
    return run_app(step=5, phq_intro_done=True, **state)


def test_the_phq8_opens_with_her_introduction_and_not_a_question():
    import html

    from auracare.phq8 import INTRO_HEADING, INTRO_SCRIPT, PHQ8_ITEMS

    at = run_app(step=5)
    markdown = " ".join(m.value for m in at.markdown)
    assert INTRO_HEADING in markdown
    # The bubble escapes its text, so the comparison has to as well.
    assert html.escape(INTRO_SCRIPT) in markdown
    assert PHQ8_ITEMS[0].text not in markdown
    assert any(b.key == "phq_begin" for b in at.button)


def test_beginning_the_questions_shows_the_first_one_only():
    from auracare.phq8 import PHQ8_ITEMS

    at = run_app(step=5).button(key="phq_begin").click().run()
    assert at.session_state["phq_intro_done"] is True
    markdown = " ".join(m.value for m in at.markdown)
    assert PHQ8_ITEMS[0].text in markdown
    for later in PHQ8_ITEMS[1:]:
        assert later.text not in markdown


def test_a_question_shows_the_clinical_wording_not_the_spoken_sentence():
    """
    
    """
    from auracare.phq8 import PHQ8_ITEMS

    at = questions()
    markdown = " ".join(m.value for m in at.markdown)
    assert PHQ8_ITEMS[0].text in markdown
    assert "You can say: not at all" not in markdown


def test_the_four_response_options_are_the_published_anchors():
    from auracare.phq8 import RESPONSE_OPTIONS

    at = questions()
    labels = [b.label for b in at.button]
    for option in RESPONSE_OPTIONS:
        assert option in labels


def test_answering_records_the_answer_and_moves_on():
    from auracare.phq8 import PHQ8_ITEMS

    at = questions()
    at = at.button(key="phq_opt_0_2").click().run()
    assert at.session_state["phq8_responses"][PHQ8_ITEMS[0].key] == 2
    assert at.session_state["phq_index"] == 1


def test_previous_goes_back_and_the_earlier_answer_is_still_marked():
    at = questions()
    at = at.button(key="phq_opt_0_3").click().run()
    at = at.button(key="phq_back").click().run()
    assert at.session_state["phq_index"] == 0
    chosen = [b for b in at.button if b.key == "phq_opt_0_3"]
    assert chosen and chosen[0].proto.type == "primary"


def test_there_is_no_previous_button_on_the_first_question():
    at = questions()
    assert not any(b.key == "phq_back" for b in at.button)


def test_all_eight_questions_are_reachable_and_the_last_one_finishes():
    from auracare.phq8 import ITEM_KEYS, PHQ8_ITEMS

    at = questions()
    for index in range(len(PHQ8_ITEMS)):
        assert at.session_state["phq_index"] == index
        at = at.button(key=f"phq_opt_{index}_1").click().run()
        assert not at.exception

    assert at.session_state["finished"] is True
    assert at.session_state["step"] == 6
    assert all(at.session_state["phq8_responses"][k] == 1 for k in ITEM_KEYS)


def test_a_question_can_be_skipped():
    """
    """
    from auracare.phq8 import PHQ8_ITEMS

    at = questions()
    at = at.button(key="phq_skip_0").click().run()
    assert at.session_state["phq8_responses"][PHQ8_ITEMS[0].key] is None
    assert at.session_state["phq_index"] == 1


def test_a_skipped_question_still_scores_by_proration():
    from auracare.phq8 import ITEM_KEYS, score_phq8

    responses = {k: 1 for k in ITEM_KEYS}
    responses[ITEM_KEYS[0]] = None
    result = score_phq8(responses)
    assert result.scorable
    assert result.prorated


def test_the_running_total_is_not_shown_while_answering():
    """
    
    """
    from auracare.phq8 import ITEM_KEYS

    at = questions(phq_index=7, phq8_responses={k: 3 for k in ITEM_KEYS})
    shown = " ".join(
        [s.value for s in at.success]
        + [c.value for c in at.caption]
        + [m.value for m in at.markdown]
    )
    assert "PHQ-8 total" not in shown
    assert "/ 24" not in shown


def test_a_stale_question_index_is_clamped_rather_than_crashing():
    at = questions(phq_index=99)
    assert not at.exception
    assert any("Question 8 of 8" in c.value for c in at.caption)


def test_starting_over_resets_the_questionnaire():
    """
    
    """
    at = questions(phq_index=5)
    reset = [b for b in at.sidebar.button if "Start over" in b.label]
    assert reset
    at = reset[0].click().run()
    assert at.session_state["phq_index"] == 0
    assert at.session_state["phq_intro_done"] is False


def test_the_introduction_works_whether_or_not_her_clip_has_arrived():
    from auracare import paths

    at = run_app(step=5)
    assert not at.exception
    expected = 1 if paths.find_animation(paths.PHQ8_INTRO_ANIMATION) else 0
    assert len(at.get("video")) == expected


def test_the_clinical_view_survives_a_session_with_nothing_measured():
    """
    """
    from auracare.phq8 import ITEM_KEYS

    at = run_app(
        step=6,
        finished=True,
        results={},
        skipped={1, 2, 3, 4},
        phq8_responses={k: 1 for k in ITEM_KEYS},
    )
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "No acoustic indicators were measured" in captions


def test_annie_speaks_the_phq8_introduction_rather_than_the_browser():
    at = run_app(step=5)
    assert not at.exception
    assert len(at.get("video")) == 1


def test_every_question_screen_plays_its_own_clip():
    """
    One audio element per question, and a different file each time. Walking
    the whole instrument is the only way to catch a clip wired to two items.
    """
    from auracare.phq8 import PHQ8_ITEMS

    urls = set()
    for index in range(len(PHQ8_ITEMS)):
        at = questions(phq_index=index)
        assert not at.exception
        players = at.get("audio")
        assert len(players) == 1, f"question {index + 1} has {len(players)}"
        urls.add(players[0].proto.url)
    assert len(urls) == len(PHQ8_ITEMS)


def test_the_browser_voice_is_not_used_when_a_clip_exists():
    """
    """
    at = questions(phq_index=0)
    assert "phq_q_0" not in at.session_state["spoken"]


# ---------------------------------------------------------------------------
# The welcome screen
# ---------------------------------------------------------------------------
def test_the_check_in_opens_on_the_welcome_screen():
    """
    """
    from auracare import ui

    at = run_app(welcome_done=False)
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert ui.WELCOME_HEADING in markdown
    assert any(b.key == "welcome_begin" for b in at.button)
    # The greeting only. No task instruction, no recorder.
    assert ui.script_for(1).instruction not in markdown
    assert not any(b.key == "skip_1" for b in at.button)


def test_only_one_clip_plays_on_the_welcome_screen():
    at = run_app(welcome_done=False)
    assert len(at.get("video")) == 1


def test_beginning_the_check_in_goes_straight_to_task_one():
    from auracare import ui

    at = run_app(welcome_done=False).button(key="welcome_begin").click().run()
    assert at.session_state["welcome_done"] is True
    assert at.session_state["step"] == 1
    markdown = " ".join(m.value for m in at.markdown)
    assert ui.script_for(1).instruction.replace("**", "") in markdown.replace("**", "")


def test_task_one_no_longer_carries_the_welcome_clip():
    """One clip on the task screen, the task's own."""
    at = run_app(step=1)
    assert len(at.get("video")) == 1


def test_starting_over_returns_to_the_welcome():
    at = run_app(step=3)
    reset = [b for b in at.sidebar.button if "Start over" in b.label]
    assert reset
    at = reset[0].click().run()
    assert at.session_state["welcome_done"] is False


# ---------------------------------------------------------------------------
# F1. The two closing clips
# ---------------------------------------------------------------------------
def _closing_video_names(at):
    return [v.proto.url for v in at.get("video")]


def _clip_asked_for(monkeypatch, **state):
    """
    
    """
    from auracare import paths

    asked = []
    real = paths.find_animation

    def spy(filename):
        asked.append(filename)
        return real(filename)

    monkeypatch.setattr(paths, "find_animation", spy)
    at = closing(**state)
    assert not at.exception
    closing_names = (
        paths.CLOSING_GOOD_ANIMATION,
        paths.CLOSING_REVIEW_ANIMATION,
        paths.CLOSING_ANIMATION,
    )
    return [name for name in asked if name in closing_names], at


def test_the_routine_ending_plays_the_clip_that_files_the_results(monkeypatch):
    """
    """
    from auracare import paths

    asked, at = _clip_asked_for(monkeypatch, **ROUTINE)
    assert asked[0] == paths.CLOSING_GOOD_ANIMATION
    assert len(_closing_video_names(at)) == 1


def test_a_clinician_review_plays_the_other_clip(monkeypatch):
    """
    
    """
    from auracare import paths

    asked, at = _clip_asked_for(monkeypatch, **REVIEW)
    assert asked[0] == paths.CLOSING_REVIEW_ANIMATION
    assert len(_closing_video_names(at)) == 1


def test_the_two_endings_do_not_play_the_same_clip():
    good = _closing_video_names(closing(**ROUTINE))
    review = _closing_video_names(closing(**REVIEW))
    assert good and review
    assert good[0] != review[0]


def test_an_emergency_ending_plays_her_emergency_clip():
    """
    
    """
    at = closing(results=emergency_results(), phq8_responses={k: 0 for k in ITEM_KEYS})
    assert not at.exception
    assert len(_closing_video_names(at)) == 1


def test_a_crisis_ending_plays_her_crisis_clip():
    """
    
    """
    at = closing(
        results=healthy_results(
            transcript_text="honestly I think I would be better off dead"
        ),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )
    assert not at.exception
    assert len(_closing_video_names(at)) == 1


# ---------------------------------------------------------------------------
# The two urgent clips: emergency supplied 18 Nov, crisis still to come
# ---------------------------------------------------------------------------
def _crisis_state():
    return dict(
        results=healthy_results(
            transcript_text="honestly I think I would be better off dead"
        ),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )


def _animations_asked_for(monkeypatch, **state):
    """Which clip filenames the closing screen looked up, and the screen."""
    from auracare import paths

    asked = []
    real = paths.find_animation

    def spy(filename):
        asked.append(filename)
        return real(filename)

    monkeypatch.setattr(paths, "find_animation", spy)
    at = closing(**state)
    assert not at.exception
    return asked, at


def test_the_emergency_ending_asks_for_the_emergency_clip(monkeypatch):
    from auracare import paths

    asked, _ = _animations_asked_for(
        monkeypatch,
        results=emergency_results(),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )
    assert any(name in paths.CLOSING_EMERGENCY_ANIMATIONS for name in asked)


def test_the_crisis_ending_asks_for_a_clip_of_its_own(monkeypatch):
    """
    
    """
    from auracare import paths

    asked, _ = _animations_asked_for(monkeypatch, **_crisis_state())
    assert any(name in paths.CLOSING_CRISIS_ANIMATIONS for name in asked)


def test_the_crisis_ending_never_borrows_the_emergency_clip(monkeypatch):
    """

    """
    from auracare import paths

    asked, at = _animations_asked_for(monkeypatch, **_crisis_state())
    assert len(_closing_video_names(at)) == 1
    assert not any(
        name in paths.CLOSING_EMERGENCY_ANIMATIONS for name in asked
    ), "the crisis ending reached for the emergency clip"


def test_a_patient_in_both_pathways_gets_the_crisis_clip_not_the_breathing_one():
    """
    Strained breathing and a disclosure of self-harm in the same check-in.
    Crisis outranks the respiratory ending, so the clip about breathing is
    not played over a banner about self-harm.
    """
    from auracare.phq8 import score_phq8
    from auracare.risk import PATHWAY_CRISIS, PATHWAY_EMERGENCY, assess_screening

    # The short phonation that trips the respiratory threshold, on top of the
    # transcript that trips the crisis one.
    both = healthy_results(
        transcript_text="honestly I think I would be better off dead"
    )
    both[1] = emergency_results()[1]
    responses = {k: 0 for k in ITEM_KEYS}

    # Graded the same way the screen grades it, so this cannot quietly become
    # a crisis-only case and go on passing for the wrong reason.
    assessment = assess_screening(
        both[1], both[2], both[3], both[4], score_phq8(responses)
    )
    assert PATHWAY_CRISIS in assessment.pathways, assessment.pathways
    assert PATHWAY_EMERGENCY in assessment.pathways, assessment.pathways

    at = closing(results=both, phq8_responses=responses)
    assert not at.exception
    # And the banner that is up is the one about self-harm.
    body = " ".join(m.value for m in at.markdown)
    assert "116 123" in body, "the crisis banner is not the one on screen"

    # The clip beside it is the crisis one, checked by content. The two are
    # separate recordings and only one of them belongs over this banner.
    import hashlib

    from auracare import paths

    played = _closing_video_names(at)
    assert len(played) == 1
    crisis = paths.find_closing_crisis()
    emergency = paths.find_closing_emergency()
    assert crisis is not None and emergency is not None
    only_crisis = closing(**_crisis_state())
    assert _closing_video_names(only_crisis) == played, (
        "the both-pathways ending played something other than the crisis clip"
    )
    assert hashlib.sha256(crisis.read_bytes()).hexdigest() != hashlib.sha256(
        emergency.read_bytes()
    ).hexdigest(), "the two urgent clips are the same recording"


def test_the_crisis_clip_plays_once_it_is_recorded(tmp_path, monkeypatch):
    """The whole point of the slot: a file copy and nothing else."""
    from auracare import paths

    monkeypatch.setattr(paths, "ANIMATIONS_DIR", tmp_path)
    (tmp_path / paths.CLOSING_CRISIS_ANIMATIONS[0]).write_bytes(b"\x00" * 64)
    at = closing(**_crisis_state())
    assert not at.exception
    assert len(_closing_video_names(at)) == 1


@pytest.mark.parametrize(
    "which", ["CLOSING_EMERGENCY_ANIMATIONS", "CLOSING_CRISIS_ANIMATIONS"]
)
def test_any_of_her_likely_filenames_is_accepted(tmp_path, monkeypatch, which):
    """
    """
    from auracare import paths

    monkeypatch.setattr(paths, "ANIMATIONS_DIR", tmp_path)
    finder = (
        paths.find_closing_emergency
        if "EMERGENCY" in which
        else paths.find_closing_crisis
    )
    for name in getattr(paths, which):
        for stale in tmp_path.iterdir():
            stale.unlink()
        (tmp_path / name).write_bytes(b"\x00" * 64)
        assert finder() is not None, name


def test_each_urgent_ending_falls_silent_if_its_own_clip_goes_missing(
    tmp_path, monkeypatch
):
    """
    The fallback that matters. An assets folder with neither urgent clip in
    it must leave both endings silent rather than reaching for whatever else
    is there.
    """
    from auracare import paths

    monkeypatch.setattr(paths, "ANIMATIONS_DIR", tmp_path)
    (tmp_path / paths.CLOSING_GOOD_ANIMATION).write_bytes(b"\x00" * 64)
    (tmp_path / paths.CLOSING_REVIEW_ANIMATION).write_bytes(b"\x00" * 64)

    assert paths.find_closing_crisis() is None
    assert paths.find_closing_emergency() is None

    at = closing(**_crisis_state())
    assert not at.exception
    assert _closing_video_names(at) == []

    at = closing(
        results=emergency_results(), phq8_responses={k: 0 for k in ITEM_KEYS}
    )
    assert not at.exception
    assert _closing_video_names(at) == []


# ---------------------------------------------------------------------------
# A skipped safety check must not read as a passed one
# ---------------------------------------------------------------------------
def _unavailable_transcript():
    from auracare.stt import Transcript

    return Transcript(
        text=None,
        available=False,
        backend="faster-whisper",
        unavailable_reason="transcription failed: model could not be loaded",
    )


def _closing_warnings(at):
    return [w.value for w in at.warning]


def test_the_ending_says_when_the_self_harm_check_did_not_run():
    """
    """
    at = closing(
        results=healthy_results(),
        transcript=_unavailable_transcript(),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )
    assert not at.exception
    warnings = " ".join(_closing_warnings(at))
    assert "could not be transcribed" in warnings
    assert "self-harm" in warnings
    assert "model could not be loaded" in warnings, (
        "the reason is not carried through"
    )


def test_a_working_transcript_produces_no_such_warning():
    at = closing(**ROUTINE)
    assert not at.exception
    assert not any(
        "could not be transcribed" in w for w in _closing_warnings(at)
    )


def test_the_warning_is_not_shown_when_task_four_was_skipped():
    """Nothing was said, so nothing failed to be understood."""
    results = healthy_results()
    del results[4]
    at = closing(
        results=results,
        transcript=_unavailable_transcript(),
        phq8_responses={k: 0 for k in ITEM_KEYS},
    )
    assert not at.exception
    assert not any(
        "could not be transcribed" in w for w in _closing_warnings(at)
    )


def test_a_crisis_ending_does_not_also_warn_that_the_check_was_skipped():
    """It plainly did run: it is why this banner is up."""
    at = closing(**_crisis_state())
    assert not at.exception
    assert not any(
        "could not be transcribed" in w for w in _closing_warnings(at)
    )


def test_the_two_urgent_endings_do_not_share_a_filename():
    """A name in both tuples would hand one clip to both endings."""
    from auracare import paths

    assert not (
        set(paths.CLOSING_EMERGENCY_ANIMATIONS)
        & set(paths.CLOSING_CRISIS_ANIMATIONS)
    )


def test_the_emergency_clip_is_not_one_of_the_other_closings():
    """
    Checked by content, not by filename, as with every clip she has sent. A
    duplicate would mean the wrong recording had been installed.
    """
    import hashlib

    from auracare import paths

    digests = set()
    for name in (paths.CLOSING_GOOD_ANIMATION, paths.CLOSING_REVIEW_ANIMATION):
        found = paths.find_animation(name)
        assert found is not None, name
        digests.add(hashlib.sha256(found.read_bytes()).hexdigest())
    emergency = paths.find_closing_emergency()
    assert emergency is not None, "her emergency clip is missing"
    assert hashlib.sha256(emergency.read_bytes()).hexdigest() not in digests


def test_the_generic_closing_clip_never_reaches_an_urgent_ending(
    tmp_path, monkeypatch
):
    """
    
    """
    from auracare import paths

    monkeypatch.setattr(paths, "ANIMATIONS_DIR", tmp_path)
    (tmp_path / paths.CLOSING_ANIMATION).write_bytes(b"\x00" * 64)

    at = closing(**_crisis_state())
    assert not at.exception
    assert _closing_video_names(at) == [], (
        "the generic closing clip played on a crisis ending"
    )

    at = closing(
        results=emergency_results(), phq8_responses={k: 0 for k in ITEM_KEYS}
    )
    assert not at.exception
    assert _closing_video_names(at) == [], (
        "the generic closing clip played on an emergency ending"
    )

    # The same fallback is still there for the endings it suits.
    at = closing(**ROUTINE)
    assert not at.exception
    assert len(_closing_video_names(at)) == 1
