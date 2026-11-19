"""
The check-in running itself: Annie first, then the recorder, then a pop-up.

Three things the clinician asked for on 18 Nov, which are really one flow:

* arriving at a task should show Annie's words, not "a brief flash of the
  next task";
* the ending affirmations should be pop-ups, so a patient scrolling down
  cannot reach the next task's controls and press one by accident;
* a finished task should give its affirmation and then start the next task
  by itself, all the way to the PHQ-8 introduction.

What cannot be asserted here is the browser half: `components.html` does not
appear in the element tree, so the script that presses a button when Annie
stops speaking is covered by `test_autoplay.py` for its markup and by the
Playwright run for its behaviour. These tests cover what Python decides.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 120

START_OVER = "\U0001f501 Start over"


def run_app(**state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    state.setdefault("welcome_done", True)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


def labels(at):
    return [b.label for b in at.button]


def press(at, label):
    for b in at.button:
        if b.label == label:
            return b.click().run()
    raise AssertionError(f"no button labelled {label!r}; saw {labels(at)}")


# ---------------------------------------------------------------------------
# Arriving at a task
# ---------------------------------------------------------------------------
def test_a_task_opens_on_annie_not_on_its_controls():
    """
    The recorder, the skip and the repeat all belong to the task. Showing
    them while Annie is still giving the instruction is what the clinician
    saw as a flash of the next task.
    """
    at = run_app(step=1, opening_done=set())
    assert "I am ready" in labels(at)
    assert "Say that again" not in labels(at)
    assert "Skip Task 1" not in labels(at)


def test_annies_words_are_on_screen_during_the_opening():
    """Her instruction is the thing that is meant to be there."""
    at = run_app(step=1, opening_done=set())
    body = " ".join(m.value for m in at.markdown)
    assert "take a deep, comfortable breath" in body


def test_i_am_ready_brings_the_recorder_up():
    at = run_app(step=1, opening_done=set())
    at = press(at, "I am ready")
    assert 1 in at.session_state.opening_done
    assert "Say that again" in labels(at)
    assert "Skip Task 1" in labels(at)


def test_the_opening_is_per_task_not_once_for_the_whole_check_in():
    """Task 2 gets its own turn to be introduced."""
    at = run_app(step=2, opening_done={1})
    assert "I am ready" in labels(at)
    assert "Say that again" not in labels(at)


def test_the_opening_does_not_hold_back_the_repeat_of_an_instruction():
    """
    CLARIFY is Annie giving the instruction again on request. The recorder
    has to be there, or "Say that again" would strand the patient on a task
    with nothing to press.
    """
    at = run_app(step=1, opening_done=set(), phase="CLARIFY", phase_step=1)
    assert "Got it, carry on" in labels(at)
    assert "I am ready" not in labels(at)


def test_the_opening_does_not_hold_back_the_quiet_prompt():
    at = run_app(step=1, opening_done=set(), phase="SILENCE_RETRY",
                 phase_step=1)
    assert "I am ready" not in labels(at)
    assert any("try again" in label for label in labels(at))


# ---------------------------------------------------------------------------
# The affirmation pop-up
# ---------------------------------------------------------------------------
def test_the_affirmation_is_a_pop_up_with_a_way_on_and_a_way_back():
    at = run_app(step=1, opening_done={1}, phase="AFFIRM", phase_step=1,
                 affirm_seen=set())
    assert "Continue to Task 2" in labels(at)
    assert "Show my measurements" in labels(at)


def test_the_pop_up_moves_on_to_the_next_task():
    at = run_app(step=1, opening_done={1}, phase="AFFIRM", phase_step=1,
                 affirm_seen=set())
    at = press(at, "Continue to Task 2")
    assert at.session_state.step == 2


def test_the_next_task_arrives_with_its_opening_still_to_come():
    """
    The point of the whole flow: moving on lands on Annie, not on a recorder.
    """
    at = run_app(step=1, opening_done={1}, phase="AFFIRM", phase_step=1,
                 affirm_seen=set())
    at = press(at, "Continue to Task 2")
    assert 2 not in at.session_state.opening_done
    assert "I am ready" in labels(at)


def test_the_pop_up_can_be_dismissed_to_read_the_measurements():
    at = run_app(step=1, opening_done={1}, phase="AFFIRM", phase_step=1,
                 affirm_seen=set())
    at = press(at, "Show my measurements")
    assert at.session_state.step == 1
    assert 1 in at.session_state.affirm_seen


def test_a_dismissed_pop_up_does_not_come_back():
    """
    Without this the dialog reopens on the next rerun of the task behind it,
    and "Show my measurements" would never reveal anything.
    """
    at = run_app(step=1, opening_done={1}, phase="AFFIRM", phase_step=1,
                 affirm_seen={1})
    assert "Show my measurements" not in labels(at)


def test_the_last_task_leads_to_the_questions_not_to_a_task_five():
    at = run_app(step=4, opening_done={4}, phase="AFFIRM", phase_step=4,
                 affirm_seen=set())
    assert "Continue to the questions" in labels(at)
    at = press(at, "Continue to the questions")
    assert at.session_state.step == 5


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------
def test_moving_on_by_itself_is_a_switch_that_starts_on():
    at = run_app(step=1)
    toggle = next(
        (t for t in at.sidebar.toggle if t.label == "Move on by itself"), None
    )
    assert toggle is not None, "no switch for the automatic flow"
    assert toggle.value is True


# ---------------------------------------------------------------------------
# Nothing is ever left with no way forward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_screen_of_the_guided_flow_offers_a_button(step):
    """
    Each state the new flow can be in has to have something to press, with
    the sidebar's own controls discounted. An automatic step that depends on
    a clip must never be the only way on, because a blocked or missing clip
    would then strand the patient.
    """
    chrome = {"Reset thresholds", START_OVER}
    for state in (
        dict(opening_done=set()),
        dict(opening_done={step}),
        dict(opening_done={step}, phase="AFFIRM", phase_step=step,
             affirm_seen=set()),
        dict(opening_done={step}, phase="CLARIFY", phase_step=step),
        dict(opening_done={step}, phase="SILENCE_RETRY", phase_step=step),
    ):
        at = run_app(step=step, **state)
        assert not at.exception, f"step {step} {state}: {at.exception}"
        assert set(labels(at)) - chrome, f"step {step} {state}: nothing to press"
