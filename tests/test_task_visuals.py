"""
Task 2's candle cue, and the drop-in slot for the clinician's own artwork.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from auracare import paths, ui


# ---------------------------------------------------------------------------
# The candles themselves
# ---------------------------------------------------------------------------
def test_task_two_shows_candles_rather_than_a_circle():
    assert ui.script_for(2).cue_class == "candles-exhale"


def test_the_other_tasks_keep_their_circles():
    """Only Task 2's instruction is about blowing something out."""
    assert ui.script_for(1).cue_class == "circle-phonation"
    assert ui.script_for(3).cue_class == "circle-ddk"


def test_the_retired_exhale_circle_is_gone_from_the_sheet():
    """Dead CSS for a cue nothing renders is a trap for the next reader."""
    assert "circle-exhale" not in ui.CSS
    assert "burst-exhale" not in ui.CSS


def test_the_cue_renders_five_candles_each_with_a_flame():
    markup = ui.cue_html(ui.script_for(2))
    assert markup.count('class="candle"') == 5
    assert markup.count('class="flame"') == 5
    assert 'class="cake-icing"' in markup
    assert 'class="cake-base"' in markup


def test_every_flame_is_inside_a_candle():
    """A flame floating on its own has nothing to be blown off."""
    markup = ui.cue_html(ui.script_for(2))
    assert '<span class="candle"><span class="flame"></span></span>' in markup
    assert markup.count("<span") == markup.count("</span>")


def test_the_flames_go_out_and_come_back():
    """
    The cue loops so it can be watched twice before recording, as the other
    two do. A cue that blew out once and stayed dark would read as finished.
    """
    block = re.search(r"@keyframes blow-out \{(.*?)^  \}", ui.CSS, re.S | re.M)
    assert block, "the blow-out keyframes are gone"
    body = block.group(1)
    assert "scale(0)" in body, "the flames never actually go out"
    assert "opacity: 0;" in body
    # Back to full size and full opacity by the end of the cycle.
    assert re.search(r"100%\s*\{ transform: scale\(1\)[^}]*opacity: 1", body)
    assert "animation: blow-out 4s infinite" in ui.CSS


def test_the_flames_go_out_in_sequence():
    """
    Five flames switching off together reads as a light being turned off.
    Staggering them reads as a breath crossing the cake, which is the
    manoeuvre being asked for.
    """
    delays = [
        float(v)
        for v in re.findall(r"\.flame \{ animation-delay: ([\d.]+)s; \}", ui.CSS)
    ]
    assert len(delays) == 4, "four of the five candles should be offset"
    assert delays == sorted(delays), "the stagger is not left to right"
    assert len(set(delays)) == 4, "two candles share a delay"
    # The whole wave has to fit inside the time the flames are lit, or the
    # last candle is relighting before the first has gone out.
    assert max(delays) < 1.0


def test_the_candles_keep_the_footprint_the_circles_have():
    """
    Moving between Task 1 and Task 2 must not move the panel. Same box, same
    reserved room as the shared circle rule.
    """
    rule = re.search(r"^  \.candles-exhale \{(.*?)^  \}", ui.CSS, re.S | re.M)
    assert rule, "the candle rule is gone"
    body = rule.group(1)
    assert "width: 160px; height: 160px;" in body
    assert "margin: 32px 0;" in body


def test_reduced_motion_leaves_the_candles_lit_rather_than_hidden():
    """
    A patient who has asked for less movement still has to see what they are
    blowing out, so the animation stops and the flames stay.
    """
    block = re.search(
        r"@media \(prefers-reduced-motion: reduce\) \{(.*?)^  \}",
        ui.CSS,
        re.S | re.M,
    )
    assert block, "the reduced-motion rule is gone"
    assert ".flame { animation: none; }" in block.group(1).replace(
        ".circle-phonation, .circle-ddk, ", ""
    )
    assert "display: none" not in block.group(1)


def test_a_cue_without_its_own_markup_still_renders_one_element():
    """The circles carry no inner parts and must not start needing any."""
    markup = ui.cue_html(ui.script_for(1))
    assert '<div class="circle-phonation"></div>' in markup


def test_task_four_still_has_no_cue_because_it_has_a_picture():
    assert ui.script_for(4).cue_class == ""
    assert ui.cue_html(ui.script_for(4)) == ""


# ---------------------------------------------------------------------------
# The drop-in slot for her own artwork
# ---------------------------------------------------------------------------
@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ASSETS_DIR", tmp_path)
    return tmp_path


def test_no_file_means_the_built_in_cue(assets):
    assert paths.find_task_visual(2) is None


@pytest.mark.parametrize(
    "name",
    ["task2_visual.mp4", "task2_visual.gif", "task2_visual.png",
     "task2_visual.webm", "task2_visual.jpeg", "task2_visual.webp"],
)
def test_any_ordinary_image_or_video_counts(assets, name):
    """
    She sends whatever her tool exports. The extension is the thing that
    varies, so it is the thing the lookup tolerates.
    """
    (assets / name).write_bytes(b"x")
    found = paths.find_task_visual(2)
    assert found is not None and found.name == name


def test_the_suffix_a_browser_adds_still_counts(assets):
    """`task2_visual (1).mp4` is what a second download is named."""
    (assets / "task2_visual (1).mp4").write_bytes(b"x")
    assert paths.find_task_visual(2) is not None


def test_the_case_of_the_name_does_not_matter(assets):
    (assets / "Task2_Visual.MP4").write_bytes(b"x")
    assert paths.find_task_visual(2) is not None


def test_the_exact_name_wins_over_a_leftover_copy(assets):
    """A stale download must not silently displace the intended file."""
    (assets / "task2_visual (1).mp4").write_bytes(b"old")
    (assets / "task2_visual.mp4").write_bytes(b"new")
    assert paths.find_task_visual(2).name == "task2_visual.mp4"


def test_one_task_s_visual_is_not_offered_to_another(assets):
    (assets / "task2_visual.mp4").write_bytes(b"x")
    assert paths.find_task_visual(1) is None
    assert paths.find_task_visual(3) is None
    assert paths.find_task_visual(4) is None


def test_an_unrelated_file_is_not_mistaken_for_one(assets):
    (assets / "stimulus_park.png").write_bytes(b"x")
    (assets / "task2_notes.txt").write_bytes(b"x")
    (assets / "task2_visual.txt").write_bytes(b"x")
    assert paths.find_task_visual(2) is None


def test_a_missing_assets_folder_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ASSETS_DIR", tmp_path / "gone")
    assert paths.find_task_visual(2) is None


def test_the_app_prefers_her_file_to_the_built_in_cue():
    """
    The cue is a stand-in for her artwork. The ordering in `app.py` is what
    makes the stand-in step aside when the artwork arrives.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    supplied = source.index("supplied = paths.find_task_visual(script.task_id)")
    cue = source.index("if script.cue_class:\n            st.markdown(ui.cue_html")
    assert supplied < cue, "the built-in cue is checked before her file"


def test_a_supplied_video_is_muted_and_looping():
    """
    It plays beside Annie, who is the one talking, and a browser will not
    autoplay a clip with sound at all.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    call = source[source.index("st.video(str(supplied)"):]
    call = call[: call.index(")\n")]
    for flag in ("autoplay=True", "loop=True", "muted=True"):
        assert flag in call, f"the supplied cue is missing {flag}"


def test_her_cake_animation_is_the_shipped_task_two_visual():
    """
    Pins the live state. Her cake arrived on 19 Nov as a GIF and went into
    the slot with no code change, which is what the slot was for. The candles
    stay in the stylesheet as the fallback for an assets folder without it.
    """
    found = paths.find_task_visual(2)
    assert found is not None, "her cake animation is missing from assets/"
    assert found.suffix.lower() == ".gif"


def test_the_shipped_cake_really_is_an_animation():
    """
    The three files she sent before this one were single still frames of a
    loading spinner, each sent believing it was the animation. Counting the
    frames is what told them apart, so it is worth counting them here too.
    """
    from PIL import Image, ImageSequence

    found = paths.find_task_visual(2)
    assert found is not None
    with Image.open(found) as im:
        frames = sum(1 for _ in ImageSequence.Iterator(im))
        assert frames > 1, f"{found.name} is a single frame, not an animation"
        assert im.info.get("loop") == 0, "the cake does not loop"


# ---------------------------------------------------------------------------
# A supplied cue must not be mistaken for Annie
# ---------------------------------------------------------------------------
def test_the_cue_container_is_keyed_so_the_scripts_can_see_it():
    """
    Streamlit's container `key` becomes a CSS class, which is the only hook
    the browser scripts have into its DOM.
    """
    assert ui.CUE_CONTAINER_CLASS == f"st-key-{ui.CUE_CONTAINER_KEY}"
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    assert "with st.container(key=ui.CUE_CONTAINER_KEY):" in source


def test_waiting_for_annie_ignores_a_supplied_cue():
    """
    Both scripts watch `video[autoplay]`, and a supplied cue is one. Verified
    in a browser on 18 Nov: with this line taken out, a two-second cue looped,
    its first `ended` counted as Annie finishing and the recorder appeared
    while she was eighteen seconds into speaking.
    """
    script = ui.advance_when_clip_ends("task1.mp4#0", "I am ready")
    assert f'.closest(".{ui.CUE_CONTAINER_CLASS}")' in script


def test_the_autoplay_nudge_ignores_a_supplied_cue():
    """
    Rendered from Annie's clip, the nudge rewinds every autoplay video whose
    nonce is not hers, which would restart a looping cue; rendered from the
    cue, it would rewind Annie mid-sentence. It is given neither job.
    """
    script = ui.autoplay_nudge("task1.mp4#0")
    assert f'.closest(".{ui.CUE_CONTAINER_CLASS}")' in script


def test_a_supplied_cue_is_not_given_a_nudge_of_its_own():
    """Muted autoplay needs no gesture, so the cue needs no nudge."""
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("supplied = paths.find_task_visual"):]
    block = block[: block.index("elif script.cue_class:")]
    assert "autoplay_nudge" not in block


def test_a_supplied_visual_keeps_the_task_label_and_hint():
    """
    The cue box carries the task's name and its one-line hint. A supplied
    visual that dropped them would leave that side of the screen saying
    nothing about what is being measured.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("supplied = paths.find_task_visual"):]
    block = block[: block.index("elif script.cue_class:")]
    assert "script.cue_label" in block
    assert "script.cue_hint" in block
    assert 'class="cue-title"' in block


# ---------------------------------------------------------------------------
# Annie heard, not operated
# ---------------------------------------------------------------------------
def test_the_voice_container_is_keyed_so_the_stylesheet_can_reach_it():
    assert ui.VOICE_CONTAINER_CLASS == f"st-key-{ui.VOICE_CONTAINER_KEY}"


def test_a_voice_clip_is_taken_out_of_sight_rather_than_removed():
    """
    `display: none` is grounds for some browsers to throttle or refuse
    playback. The element stays laid out and playing; it is just not drawn.
    """
    rule = re.search(
        r"\.st-key-annie-voice \[data-testid=\"stAudio\"\] \{(.*?)\}",
        ui.CSS,
        re.S,
    )
    assert rule, "the voice rule is gone"
    body = rule.group(1)
    assert "display: none" not in body
    assert "opacity: 0" in body
    assert "position: absolute" in body


def test_the_control_comes_back_when_the_clip_will_not_start():
    """
    Silence with nothing to press is worse than a player bar, so the hiding
    is undone by a class the browser adds when playback is refused.
    """
    rule = re.search(
        r"\.st-key-annie-voice\.auracare-needs-tap "
        r"\[data-testid=\"stAudio\"\] \{(.*?)\}",
        ui.CSS,
        re.S,
    )
    assert rule, "there is no way back to the control"
    assert "opacity: 1" in rule.group(1)
    assert "position: static" in rule.group(1)
    assert "Press play to hear Annie." in ui.CSS


def test_the_nudge_finds_a_voice_clip_by_its_container_too():
    """
    Streamlit drops `autoplay` from an element it has already rendered, and
    the quiet prompt lives in a fragment that re-renders every two seconds,
    so matching on the attribute alone loses it.
    """
    script = ui.autoplay_nudge("quiet_retry.mp3#1")
    assert f".{ui.VOICE_CONTAINER_CLASS} audio" in script


def test_a_refused_voice_clip_arms_the_next_touch():
    """
    Safari refuses audio on render even after a click elsewhere, which is
    what her 19 Nov screenshot shows. It does allow play() from inside a
    gesture handler, so the next touch anywhere has to retry.
    """
    script = ui.autoplay_nudge("quiet_retry.mp3#1")
    assert 'addEventListener("pointerdown"' in script
    assert 'addEventListener("keydown"' in script
    assert f'classList.add("{ui.VOICE_NEEDS_TAP_CLASS}")' in script
    assert f'classList.remove("{ui.VOICE_NEEDS_TAP_CLASS}")' in script


def test_the_quiet_prompt_and_the_clarification_are_the_discreet_ones():
    """
    Her words: hide the playbacks for silence, retries and clarification.
    Not the prompt or the affirmation, where Annie is on screen.
    """
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    assert "play_clip(flow.QUIET_RETRY_AUDIO, discreet=True)" in source
    assert "play_clip(media.clarify, discreet=True)" in source
    # The clarification fallback is one of Annie's videos, so she is visible
    # and the player belongs to the picture.
    assert "play_clip(script.animation)" in source


def test_a_discreet_clip_still_gets_its_nudge():
    """Hidden and silent would be worse than visible and silent."""
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("def play_clip("):]
    block = block[: block.index("\n\n\n")]
    assert "with st.container(key=ui.VOICE_CONTAINER_KEY):" in block
    assert "autoplay_nudge" in block


def test_the_returned_control_has_a_width():
    """
    An <audio> element is inline by default, so `width: auto` leaves it zero
    pixels wide: a control that is technically back and practically still
    missing. Measured at zero in Chromium on 19 Nov before this was pinned.
    """
    rule = re.search(
        r"\.st-key-annie-voice\.auracare-needs-tap "
        r"\[data-testid=\"stAudio\"\] \{(.*?)\}",
        ui.CSS,
        re.S,
    )
    assert rule, "the revealed rule is gone"
    # Comments out first: this rule explains itself by naming the declaration
    # it must not carry, and matching that sentence is not a finding.
    body = re.sub(r"/\*.*?\*/", "", rule.group(1), flags=re.S)
    assert "width: 100%" in body
    assert "width: auto" not in body
    assert "display: block" in body
    # And a height. The hidden rule gives the element one, so the revealed
    # rule has to take it back; `height: auto` alone measured zero pixels in
    # Chromium, which is a control that is back in the layout and still
    # invisible.
    assert "min-height" in body
