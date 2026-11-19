"""
Annie's clips have to start on their own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from auracare import ui  # noqa: E402


# ---------------------------------------------------------------------------
# The markup
# ---------------------------------------------------------------------------
def test_nudge_calls_play():
    assert ".play()" in ui.autoplay_nudge("task1.mp4")


def test_nudge_reaches_the_parent_document():
    """The script runs inside a components iframe, not beside the video."""
    assert "window.parent" in ui.autoplay_nudge("task1.mp4")


def test_nudge_differs_per_clip():
    """
    The first mistake. `components.html` is served as an iframe carrying the
    markup in `srcdoc`; identical markup at the same position is reused on a
    rerun and the script never runs again. A constant string nudged the task
    prompt and then sat inert while the affirmation rendered, so the
    affirmation was silent - exactly the "missing affirmations" report.
    """
    prompt = ui.autoplay_nudge("task1.mp4")
    affirm = ui.autoplay_nudge("task1_affirm.mp4")
    quiet = ui.autoplay_nudge("quiet_retry.mp3")
    assert prompt != affirm
    assert affirm != quiet
    assert prompt != quiet


def test_nudge_embeds_the_clip_name():
    assert "task1_affirm.mp4" in ui.autoplay_nudge("task1_affirm.mp4")


def test_nudge_keeps_polling_after_finding_a_clip():
    """
    The second mistake. The first version stopped retrying as soon as any
    media element was on the page. On a phase change the script renders before
    the source has been swapped, so it found the *previous* clip - already
    nudged - and gave up, and the affirmation was still silent.
    """
    script = ui.autoplay_nudge("task1_affirm.mp4")
    assert "if (--attempts > 0)" in script
    assert "if (!found && --attempts > 0)" not in script


def test_nudge_does_not_restart_a_clip_it_already_started():
    """Otherwise every rerun would replay Annie from the top."""
    assert "dataset.auracareNudged" in ui.autoplay_nudge("task1.mp4")


def test_nudge_leaves_a_deliberately_paused_clip_alone():
    script = ui.autoplay_nudge("task1.mp4")
    assert "m.ended" in script
    assert "m.currentTime > 0.05" in script


def test_nudge_only_touches_clips_meant_to_autoplay():
    script = ui.autoplay_nudge("task1.mp4")
    assert "video[autoplay], audio[autoplay]" in script


def test_nudge_quotes_the_clip_name():
    """A filename with a quote in it must not break out of the script."""
    assert '\\"' in ui.autoplay_nudge('od"d.mp4')


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------
@pytest.fixture()
def spy(monkeypatch):
    """Capture what play_clip renders, without a browser."""
    seen = {"video": [], "audio": [], "nudges": []}
    monkeypatch.setattr(app.st, "video",
                        lambda src, **kw: seen["video"].append((src, kw)))
    monkeypatch.setattr(app.st, "audio",
                        lambda src, **kw: seen["audio"].append((src, kw)))
    monkeypatch.setattr(app.components, "html",
                        lambda html, **kw: seen["nudges"].append(html))
    return seen


def test_video_clip_is_nudged(spy):
    assert app.play_clip("task1.mp4") is True
    assert len(spy["video"]) == 1
    assert len(spy["nudges"]) == 1
    assert "task1.mp4" in spy["nudges"][0]


def test_audio_clip_is_nudged(spy):
    assert app.play_clip("quiet_retry.mp3") is True
    assert len(spy["audio"]) == 1
    assert len(spy["nudges"]) == 1
    assert "quiet_retry.mp3" in spy["nudges"][0]


def test_affirmation_clips_are_nudged(spy):
    """The clips the clinician reported as missing."""
    for step in (1, 2, 3, 4):
        spy["nudges"].clear()
        name = f"task{step}_affirm.mp4"
        assert app.play_clip(name) is True, f"{name} is not installed"
        assert len(spy["nudges"]) == 1
        assert name in spy["nudges"][0]


def test_nudge_is_per_clip_in_practice(spy):
    """Two clips in one rerun must not produce the same iframe markup."""
    app.play_clip("task1.mp4")
    app.play_clip("task1_affirm.mp4")
    assert len(spy["nudges"]) == 2
    assert spy["nudges"][0] != spy["nudges"][1]


def test_a_clip_that_is_not_autoplaying_is_not_nudged(spy):
    assert app.play_clip("task1.mp4", autoplay=False) is True
    assert spy["nudges"] == []


def test_a_missing_clip_renders_nothing(spy):
    assert app.play_clip("there_is_no_such_clip.mp4") is False
    assert spy["video"] == []
    assert spy["nudges"] == []

# ---------------------------------------------------------------------------
# Repeating a clip on purpose
# ---------------------------------------------------------------------------
def test_the_guard_is_keyed_on_the_nonce_not_the_file():
    """
    "Say that again" on a task with no clarification clip of its own replays
    the prompt clip, the same file that has just finished. Keyed on the file,
    the nudge saw a clip it had already started and left it silent: the
    subtitle changed and Annie said nothing, which is what the clinician
    reported on 17 Nov.
    """
    script = ui.autoplay_nudge("task1.mp4#2")
    assert "m.dataset.auracareNudged === clip" in script
    assert "m.dataset.auracareNudged === src" not in script


def test_a_repeated_clip_is_rewound_before_it_is_played():
    """A clip sitting at its end plays nothing unless it is wound back."""
    script = ui.autoplay_nudge("task1.mp4#2")
    assert "m.currentTime = 0" in script


def test_the_nonce_changes_when_the_conversation_moves_on(spy):
    """Two renderings of one file, separated by a phase change, must differ."""
    app.play_clip("task1.mp4")
    app.bump_clip_epoch()
    app.play_clip("task1.mp4")
    assert len(spy["nudges"]) == 2
    assert spy["nudges"][0] != spy["nudges"][1]


def test_the_nonce_holds_still_when_the_page_merely_redraws(spy):
    """
    Streamlit reruns the whole script on any interaction. If that alone
    changed the nonce, Annie would restart mid-sentence whenever the patient
    touched a control.
    """
    app.play_clip("task1.mp4")
    app.play_clip("task1.mp4")
    assert spy["nudges"][0] == spy["nudges"][1]


def test_clip_nonce_carries_both_the_file_and_the_epoch():
    from pathlib import Path

    before = app.clip_nonce(Path("task1.mp4"))
    app.bump_clip_epoch()
    after = app.clip_nonce(Path("task1.mp4"))
    assert before.startswith("task1.mp4#")
    assert before != after

# ---------------------------------------------------------------------------
# Moving on when Annie stops speaking
# ---------------------------------------------------------------------------
def test_the_advance_script_waits_for_the_clip_rather_than_a_timer():
    """
    A timer would need every clip's length written down and would drift the
    moment one is re-recorded. The element already knows when it has ended.
    """
    script = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    assert 'addEventListener("ended"' in script
    assert "m.duration" not in script


def test_the_advance_script_presses_a_real_button():
    """
    Every automatic step has to be a button the patient could press, so a
    blocked or missing clip cannot strand anyone.
    """
    script = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    assert "Continue to Task 2" in script
    assert "b.click()" in script


def test_the_advance_script_stays_inside_an_open_dialog():
    """
    The task behind the pop-up carries a button with the same wording.
    Pressing that one would advance the check-in underneath the dialog and
    leave it on screen.
    """
    script = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    assert '[role="dialog"]' in script
    assert "const root = dialog || doc;" in script


def test_the_advance_script_presses_once():
    """A rerun must not fire a second advance and skip a task."""
    script = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    assert "dataset.auracareAdvanced" in script


def test_a_clip_that_already_ended_still_counts():
    """
    The script is inserted after the clip, so a short affirmation can finish
    before the listener is attached.
    """
    script = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    assert "if (m.ended) { press(); return; }" in script


def test_each_rendering_gets_its_own_iframe():
    """Identical srcdoc is reused across reruns and never runs again."""
    a = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    b = ui.advance_when_clip_ends("task1_affirm.mp4#4", "Continue to Task 2")
    assert a != b


def test_the_label_is_escaped_into_the_script():
    """A label is interpolated, so it goes through json.dumps, not quotes."""
    script = ui.advance_when_clip_ends("x#1", 'He said "go"')
    assert '"He said \\"go\\""' in script

