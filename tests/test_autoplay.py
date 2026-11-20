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
    assert "if (m.ended) { if (!seen) press(); return; }" in script


def test_a_leftover_ended_clip_does_not_press_the_button():
    """
    Only the first script to watch an element may act on finding it already
    ended. An element carrying an earlier clip's mark is the previous phase's
    clip still on the page while Streamlit swaps it out; pressing for that
    one skips the instruction the patient has not heard yet, which is the
    "it moves on before Annie speaks" half of the 19 Nov report.
    """
    script = ui.advance_when_clip_ends("task2.mp4#7", "I am ready")
    assert "const seen = m.dataset.auracareWatched;" in script
    # The mark is still written, so a second script for the same clip is a
    # no-op either way.
    assert "m.dataset.auracareWatched = clip;" in script


def test_the_nudge_starts_the_clip_it_was_rendered_beside():
    """
    `play_clip` writes the media element and then this iframe, so the clip
    this script belongs to is the one directly above it. Finding it by
    position is what stops a leftover being restarted: a document-wide search
    also matches the previous phase's clip during the frames where Streamlit
    still has it on the page, and since the nonce has moved on, the
    per-element guard reads that leftover as a clip nobody has started yet.
    The quiet prompt was rewound and played over the top of a new task, just
    before Annie spoke, which the clinician reported on 19 Nov.
    """
    script = ui.autoplay_nudge("task2.mp4#7")
    assert "window.frameElement" in script
    assert "previousElementSibling" in script
    assert "const own = ownClip();" in script


def test_the_document_wide_sweep_cannot_restart_a_leftover():
    """
    The sweep is kept as a fallback for a Streamlit that wraps components
    differently, but it may only touch a clip nothing has nudged yet. A
    leftover always carries an earlier nonce, so it is never eligible.
    """
    script = ui.autoplay_nudge("task2.mp4#7")
    sweep = script[script.index("doc.querySelectorAll("):]
    assert "if (!m.dataset.auracareNudged) start(m);" in sweep


def test_a_deliberate_repeat_still_restarts_the_same_file():
    """
    "Say that again" on a task with no clarification clip replays the prompt
    clip, the same file that has just finished. The guard is keyed on the
    rendering rather than the file so that still rewinds and plays.
    """
    script = ui.autoplay_nudge("task1.mp4#5")
    assert "if (m.dataset.auracareNudged === clip) return;" in script
    assert "m.currentTime = 0;" in script


def test_each_rendering_gets_its_own_iframe():
    """Identical srcdoc is reused across reruns and never runs again."""
    a = ui.advance_when_clip_ends("task1_affirm.mp4#3", "Continue to Task 2")
    b = ui.advance_when_clip_ends("task1_affirm.mp4#4", "Continue to Task 2")
    assert a != b


def test_the_label_is_escaped_into_the_script():
    """A label is interpolated, so it goes through json.dumps, not quotes."""
    script = ui.advance_when_clip_ends("x#1", 'He said "go"')
    assert '"He said \\"go\\""' in script



# ---------------------------------------------------------------------------
# Telling the server that Annie has actually started
# ---------------------------------------------------------------------------
def test_the_clip_start_watcher_only_watches_this_rendering():
    """
    Matched by the mark `autoplay_nudge` leaves, which carries this exact
    nonce. A leftover from the previous phase has an older one and is
    ignored - the same trap that had the nudge replaying old clips.
    """
    script = ui.press_when_clip_plays("task2.mp4#7", ui.SPOKE_CONTAINER_CLASS)
    assert 'm.dataset.auracareNudged !== clip' in script
    assert '"task2.mp4#7"' in script


def test_the_clip_start_watcher_ignores_the_cue_and_the_quiet_clip():
    """
    The pacing visual beside Annie is not Annie, and the quiet prompt's own
    clip is not the instruction.
    """
    script = ui.press_when_clip_plays("task1.mp4#1", ui.SPOKE_CONTAINER_CLASS)
    assert '.closest(".st-key-task-cue")' in script
    assert '.closest(".st-key-annie-voice-quiet")' in script


def test_a_clip_already_finished_counts_as_given():
    """
    An instruction heard in full during the opening has been given, whatever
    the element does afterwards. `currentTime` rather than the `play` event
    for the same reason: a clip already playing when this script is inserted
    never fires `play` again.
    """
    script = ui.press_when_clip_plays("task1.mp4#1", ui.SPOKE_CONTAINER_CLASS)
    assert "m.currentTime > 0.2 || m.ended" in script


def test_the_clip_start_watcher_presses_only_once_per_clip():
    script = ui.press_when_clip_plays("task1.mp4#1", ui.SPOKE_CONTAINER_CLASS)
    assert "dataset.auracareSpoke" in script


def test_the_clip_start_watcher_targets_its_own_container():
    script = ui.press_when_clip_plays("task1.mp4#1", ui.SPOKE_CONTAINER_CLASS)
    assert f'".{ui.SPOKE_CONTAINER_CLASS}"' in script


def test_the_reporting_control_is_out_of_the_tab_order():
    """
    `visibility: hidden` rather than the one-pixel trick the clips use: it
    keeps the button off the keyboard path, which a one-pixel button is not.
    The clips cannot use it, because a browser may treat an invisible media
    element as grounds to throttle playback.
    """
    import re

    rule = re.search(r"\.st-key-annie-spoke \{(.*?)\}", ui.CSS, re.S)
    assert rule, "no rule for the reporting control"
    assert "visibility: hidden" in rule.group(1)
