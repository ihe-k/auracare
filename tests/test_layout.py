"""
The four presentation fixes the clinician asked for on 17 Nov.

Each one is small and each one was reported from a photograph, so each is
pinned here rather than left to the next screenshot.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from auracare import ui  # noqa: E402

TITLE = "Auracare: Multimodal Vocal & Cognitive Health Screening App"


# ---------------------------------------------------------------------------
# The title
# ---------------------------------------------------------------------------
def test_page_heading_is_the_title_she_asked_for():
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert 'st.title("' + chr(92) + 'U0001fa7a ' + TITLE + '")' in source


def test_browser_tab_carries_the_same_title():
    """A tab still reading 'voice screening' would contradict the page."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert f'page_title="{TITLE}"' in source
    # Every title this app has carried, so a stale one cannot survive in
    # the tab while the page shows the current one. She has renamed it
    # three times; each rename leaves its predecessor here.
    for superseded in ("Auracare voice screening",
                       "Auracare: Guided Vocal Biomarkers Screening"):
        assert superseded not in source, f'{superseded!r} is still in app.py'


# ---------------------------------------------------------------------------
# The circle growing over the sentence beneath it
# ---------------------------------------------------------------------------
def _circle_rule() -> str:
    """The shared sizing rule for the circle cues, whichever still exist.

    Written to the selector rather than from it, so retiring one circle does
    not take the arithmetic below with it: Task 2's circle became her candles
    on 18 Nov and this went on measuring the two that remain.
    """
    match = re.search(
        r"^  ((?:\.circle-[\w-]+(?:, )?)+) \{(.*?)\}", ui.CSS, re.S | re.M
    )
    assert match, "the shared circle rule is gone"
    return match.group(2)


def _circle_animation_names() -> list:
    """The keyframes the circle cues actually run."""
    names = []
    for selector in re.findall(
        r"^  (\.circle-[\w-]+) \{(.*?)\}", ui.CSS, re.S | re.M
    ):
        found = re.search(r"animation: ([\w-]+) ", selector[1])
        if found:
            names.append(found.group(1))
    assert names, "no circle animations found"
    return names


def test_the_circle_reserves_room_for_the_size_it_grows_to():
    """
    `transform: scale()` paints outside the element and reserves nothing, so
    the circle grew over the sentence under it. The reserved 32px was sized
    against the exhale burst at 1.3, the widest the sheet ever ran; the burst
    retired with Task 2's circle on 18 Nov and the margin is kept, because
    `.candles-exhale` matches it so the panel does not move between tasks.
    """
    assert "margin: 32px 0;" in _circle_rule()


def test_the_reserved_room_still_covers_the_widest_animation():
    """If a keyframe is ever scaled up, this is what notices."""
    # Scoped to the keyframes the circles run. Scanning the whole sheet, which
    # is what this did, measured Task 2's candle flames against the circle's
    # width once the candles arrived: two unrelated elements, one sum.
    scales = []
    for name in _circle_animation_names():
        block = re.search(
            r"@keyframes %s \{(.*?)^  \}" % name, ui.CSS, re.S | re.M
        )
        assert block, f"@keyframes {name} is missing"
        scales += [
            float(v)
            for v in re.findall(r"transform: scale\(([\d.]+)\)", block.group(1))
        ]
    assert scales, "no scaled keyframes found"
    width = float(re.search(r"width: (\d+)px", _circle_rule()).group(1))
    overshoot = (width * max(scales) - width) / 2
    reserved = float(re.search(r"margin: (\d+)px 0;", _circle_rule()).group(1))
    assert reserved >= overshoot, (
        f"the circle grows {overshoot:.0f}px past its box but only "
        f"{reserved:.0f}px is reserved"
    )
    # Clearing the overlap is not enough by itself. She reported the sentence
    # as crowded, not as covered, so there has to be room left over.
    assert reserved - overshoot >= 6, "no breathing room beyond the overlap"


def test_the_sentence_keeps_a_gap_of_its_own():
    assert ".cue-title { margin-top: 12px;" in ui.CSS


# ---------------------------------------------------------------------------
# The PHQ-8 cut-points in the sidebar
# ---------------------------------------------------------------------------
def test_cut_point_numbers_share_one_right_edge():
    rule = re.search(r"\.cut-row \{(.*?)\}", ui.CSS, re.S)
    assert rule, "the cut-point row rule is gone"
    assert "justify-content: space-between" in rule.group(1)


def test_cut_point_numbers_use_tabular_figures():
    """So 5 and 10 and 15 line up with each other, not just with the edge."""
    assert "font-variant-numeric: tabular-nums" in ui.CSS


def test_cut_points_are_no_longer_padded_text():
    """
    The old line padded the label to 24 characters, which lines numbers up in
    a monospace font and leaves them ragged in the app's proportional one.
    That is what she photographed.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "{key.replace('_', ' '):<24}" not in source
    assert 'class="cut-row"' in source

def test_cut_point_rows_follow_the_theme():
    """
    Fixed slate greys suit the dark panels elsewhere in the sheet and all but
    vanish in the sidebar under a light theme. `st.text`, which this replaced,
    followed the theme, so the replacement has to as well.
    """
    rule = re.search(r"\.cut-row \{(.*?)\}", ui.CSS, re.S).group(1)
    assert "color: inherit" in rule
    assert "#cbd5e1" not in rule and "#e2e8f0" not in rule

# ---------------------------------------------------------------------------
# The clinical view
# ---------------------------------------------------------------------------
def test_no_resource_tally_under_the_fhir_heading():
    """
    She asked on 17 Nov for the line under "Simulated FHIR dispatch" to go.
    It read "1x CommunicationRequest . 1x QuestionnaireResponse . ... all
    tagged HTEST".
    """
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "resources, all tagged HTEST" not in source
    assert 'st.markdown("#### Simulated FHIR dispatch")' in source


def test_the_bundle_summary_still_exists_for_the_tests():
    """Removing the caption must not remove the function behind it."""
    from auracare.fhir import summarise_bundle

    assert callable(summarise_bundle)

