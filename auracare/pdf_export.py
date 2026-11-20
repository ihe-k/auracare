"""
The patient's takeaway, as a PDF they can keep.



*No new dependency.* The pages are drawn with matplotlib, which is already
pinned for the plots. `reportlab` would have been the obvious reach and would
have meant another wheel in `requirements.txt` for a one-page document.
Nothing here needs a network, so a machine with no internet still produces the
file; `jspdf` from a CDN would not have.

*The honesty text is not optional.* Everything the app says about itself on
screen - prototype, not a diagnosis, provisional thresholds, nothing
transmitted - is repeated on page one. A screenshot of a result can be
explained by the screen around it. A PDF cannot: it gets printed, forwarded
and read a month later by someone who never saw the app, so it has to carry
its own caveats.

*A crisis pathway puts the helplines on the page.* If the check-in reached the
crisis or emergency pathway, `CRISIS_RESOURCES_UK` is printed in full rather
than referred to. A patient who needs that number should not have to go back
into the app to find it.
"""
from __future__ import annotations

import base64
import datetime as dt
import html as _html
import io
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .phq8 import MAX_SCORE
from .risk import (
    CRISIS_RESOURCES_UK,
    NOT_MEASURED,
    PATHWAY_CRISIS,
    PATHWAY_EMERGENCY,
    PATHWAY_LABEL,
)

__all__ = [
    "DISCLAIMER",
    "generate_diagnostic_summary",
    "get_pdf_download_component",
    "mailto_link",
    "render_pdf",
    "summary_filename",
    "summary_as_text",
    "summary_subject",
]

# Said on screen and said again here. See the module docstring for why this is
# repeated rather than referenced.
DISCLAIMER = (
    "This is a screening prototype, not a medical device and not a diagnosis. "
    "The voice measures are provisional engineering values, tunable in the "
    "app and deliberately separate from published clinical reference values. "
    "The questionnaire score is scored against published cut-points. Nothing "
    "in this check-in was transmitted anywhere, and no clinical decision "
    "should rest on this document alone."
)

# What each of Task 1's three numbers is, in words a patient can read. The
# ranges are the app's own provisional operating points, named as such.
_VOICE_LABELS = {
    "jitter_local_pct": ("Pitch steadiness (jitter)", "%"),
    "shimmer_local_pct": ("Loudness steadiness (shimmer)", "%"),
    "hnr_db": ("Clarity over background noise (HNR)", "dB"),
}

_VOICE_ALIASES = {
    "jitter": "jitter_local_pct",
    "shimmer": "shimmer_local_pct",
    "hnr": "hnr_db",
}

_PAGE_W_IN = 8.27          # A4 portrait
_PAGE_H_IN = 11.69
_MARGIN_IN = 0.85

_INK = "#15213a"
_MUTED = "#5b6780"
_RULE = "#c8d0e0"
_ALERT = "#b3261e"
_OK = "#1d6f42"


# ---------------------------------------------------------------------------
# Building the payload
# ---------------------------------------------------------------------------
def _f(value: Any) -> Optional[float]:
    """A float, or None for anything that is not one. Never raises."""
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN is not a measurement


def _phq8_block(phq8_responses: Any, assessment: Any) -> Dict[str, Any]:
    """
    The questionnaire result, preferring the assessment's own scoring.

    The assessment has already scored it, including the proration rule for a
    partly answered form. Re-scoring here from the raw answers would be a
    second implementation of the same rule, and the two would drift.
    """
    scored = getattr(assessment, "phq8", None) if assessment is not None else None
    if scored is not None:
        return {
            "scorable": bool(getattr(scored, "scorable", False)),
            "score": getattr(scored, "total_score", None),
            "max": MAX_SCORE,
            "band": getattr(scored, "band", None),
            "n_answered": getattr(scored, "n_answered", None),
            "prorated": bool(getattr(scored, "prorated", False)),
        }

    answers = [
        v for v in (phq8_responses or {}).values() if _f(v) is not None
    ] if isinstance(phq8_responses, Mapping) else []
    return {
        "scorable": False,
        "score": None,
        "max": MAX_SCORE,
        "band": None,
        "n_answered": len(answers),
        "prorated": False,
    }


def _repoint(message: str) -> str:
    """
    Re-aim a message written for the screen at this page's layout.

    `assess_screening` ends its severe-band message with "reach out to one of
    the services below", which is true of the Results tab, where the helplines
    follow. Here they are boxed near the top, above the questionnaire, because
    a number someone may need should not be the last bullet of a one-page
    document. Pointing the reader downwards at something printed above them is
    the kind of small wrongness that makes a patient doubt the rest of it.
    """
    return str(message).replace(
        "one of the services below",
        "the numbers at the top of this page",
    )


def _task_measured(assessment: Any, task_id: int) -> bool:
    for task in getattr(assessment, "tasks", []) or []:
        if getattr(task, "task_id", None) == task_id:
            return bool(getattr(task, "measured", False))
    return False


def _task_blocks(assessment: Any) -> List[Dict[str, Any]]:
    if assessment is None:
        return []
    out: List[Dict[str, Any]] = []
    for task in getattr(assessment, "tasks", []) or []:
        out.append({
            "task_id": getattr(task, "task_id", None),
            "title": getattr(task, "title", ""),
            "level": getattr(task, "level", NOT_MEASURED),
            "measured": bool(getattr(task, "measured", False)),
            "indicators": [
                {
                    "label": getattr(i, "label", ""),
                    "value": _f(getattr(i, "value", None)),
                    "unit": getattr(i, "unit", ""),
                    "level": getattr(i, "level", NOT_MEASURED),
                }
                for i in (getattr(task, "indicators", []) or [])
            ],
        })
    return out


def generate_diagnostic_summary(
    phq8_responses: Any = None,
    voice_metrics: Any = None,
    *,
    patient_id: Optional[str] = None,
    assessment: Any = None,
    generated_at: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """
    Everything the takeaway needs, as a plain dictionary.
    """
    raw = voice_metrics if isinstance(voice_metrics, Mapping) else {}
    metrics = {_VOICE_ALIASES.get(k, k): v for k, v in raw.items()}
    # The three numbers come from Task 1. If the assessment says Task 1 was
    # not measured, they cannot be from this run, and a page that lists the
    # task as unmeasured and then prints its jitter contradicts itself in
    # front of the patient. Dropped rather than reconciled.
    if assessment is not None and not _task_measured(assessment, 1):
        metrics = {}
    pathways = list(getattr(assessment, "pathways", []) or [])
    primary = getattr(assessment, "primary_pathway", None)
    crisis_pathway = bool(
        {PATHWAY_CRISIS, PATHWAY_EMERGENCY} & set(pathways)
        or primary in (PATHWAY_CRISIS, PATHWAY_EMERGENCY)
    )
    messages = [m for m in (getattr(assessment, "patient_messages", []) or []) if m]
    # The crisis *pathway* only fires on a self-harm disclosure in the Task 4
    # transcript. A severe PHQ-8 band does not raise it, but `assess_screening`
    # still appends the helplines to the patient messages, where on this page
    # they would be the fourth bullet of a closing section. Whenever the app
    # has decided the patient should see that number, the page gives it a box
    # of its own near the top - and then drops it from the bullets, so it is
    # printed once rather than twice. Read off the app's own decision rather
    # than re-testing the band here, so there is one rule, not two.
    needs_help_now = crisis_pathway or CRISIS_RESOURCES_UK in messages
    if needs_help_now:
        messages = [_repoint(m) for m in messages]
    when = generated_at or dt.datetime.now()
    return {
        "patient_id": (patient_id or "").strip() or None,
        "generated_at": when.replace(microsecond=0).isoformat(),
        "phq8": _phq8_block(phq8_responses, assessment),
        "voice": {
            key: _f(metrics.get(key)) for key in _VOICE_LABELS
        },
        "tasks": _task_blocks(assessment),
        "n_tasks_measured": getattr(assessment, "n_tasks_measured", None),
        "overall_level": getattr(assessment, "overall_level", NOT_MEASURED),
        "pathway": primary,
        "pathway_label": PATHWAY_LABEL.get(primary, None) if primary else None,
        "patient_messages": [m for m in messages if m != CRISIS_RESOURCES_UK],
        "crisis": needs_help_now,
        "crisis_pathway": crisis_pathway,
        "crisis_resources": CRISIS_RESOURCES_UK if needs_help_now else None,
        "disclaimer": DISCLAIMER,
    }


def summary_filename(summary: Mapping[str, Any]) -> str:
    """A filename that sorts by date and says whose it is."""
    stamp = str(summary.get("generated_at") or "")[:10] or "undated"
    who = str(summary.get("patient_id") or "patient")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in who)
    return f"auracare-summary-{safe}-{stamp}.pdf"


# ---------------------------------------------------------------------------
# Drawing it
# ---------------------------------------------------------------------------
class _Sheet:
    """
    A cursor that writes down an A4 page and starts a new one when it runs out.

    Figure coordinates throughout, so nothing depends on an axes being there.
    Wrapping is by character count against a measured average width rather
    than by rendered extent: a takeaway is body text in one size, and asking
    the renderer for the width of every line to save a few characters per line
    is not worth the slower path.
    """

    def __init__(self, title: str) -> None:
        from matplotlib.figure import Figure

        self._Figure = Figure
        self._title = title
        self.pages: List[Any] = []
        self._new_page()

    def _new_page(self) -> None:
        fig = self._Figure(figsize=(_PAGE_W_IN, _PAGE_H_IN), dpi=100)
        fig.patch.set_facecolor("white")
        self.fig = fig
        self.pages.append(fig)
        self.y = 1.0 - (_MARGIN_IN / _PAGE_H_IN)
        if len(self.pages) > 1:
            self.para(f"{self._title} (continued)", size=8, colour=_MUTED)
            self.rule()

    # -- geometry ---------------------------------------------------------
    @property
    def _left(self) -> float:
        return _MARGIN_IN / _PAGE_W_IN

    @property
    def _width(self) -> float:
        return 1.0 - 2 * (_MARGIN_IN / _PAGE_W_IN)

    def _line_h(self, size: float) -> float:
        return (size * 1.45) / 72.0 / _PAGE_H_IN

    def _chars(self, size: float, indent_in: float = 0.0) -> int:
        usable_in = (_PAGE_W_IN - 2 * _MARGIN_IN) - indent_in
        return max(20, int(usable_in * 72.0 / (size * 0.50)))

    def _room(self, needed: float) -> None:
        if self.y - needed < (_MARGIN_IN / _PAGE_H_IN):
            self._new_page()

    # -- marks ------------------------------------------------------------
    def space(self, inches: float = 0.12) -> None:
        self.y -= inches / _PAGE_H_IN

    def rule(self, colour: str = _RULE) -> None:
        from matplotlib.lines import Line2D

        self._room(0.02)
        self.fig.add_artist(Line2D(
            [self._left, self._left + self._width], [self.y, self.y],
            color=colour, linewidth=0.8, transform=self.fig.transFigure,
        ))
        self.space(0.10)

    def para(
        self, text: str, *, size: float = 9.5, colour: str = _INK,
        weight: str = "normal", indent_in: float = 0.0,
    ) -> None:
        if not text:
            return
        lines: List[str] = []
        for block in str(text).split("\n"):
            lines.extend(textwrap.wrap(block, self._chars(size, indent_in))
                         or [""])
        for line in lines:
            self._room(self._line_h(size))
            self.fig.text(
                self._left + indent_in / _PAGE_W_IN, self.y, line,
                size=size, color=colour, weight=weight,
                va="top", ha="left", wrap=False,
            )
            self.y -= self._line_h(size)

    def heading(self, text: str) -> None:
        self.space(0.14)
        self._room(self._line_h(11.5) * 2)
        self.para(text, size=11.5, weight="bold")
        self.space(0.04)

    def bullet(self, text: str) -> None:
        self._room(self._line_h(9.5))
        self.fig.text(self._left, self.y, "•", size=9.5, color=_MUTED,
                      va="top", ha="left")
        self.para(text, indent_in=0.18)

    def row(self, left: str, right: str, *, colour: str = _INK) -> None:
        """A label and its value, the value right-aligned against the margin."""
        self._room(self._line_h(9.5))
        top = self.y
        self.fig.text(self._left, top, left, size=9.5, color=_INK,
                      va="top", ha="left")
        self.fig.text(self._left + self._width, top, right, size=9.5,
                      color=colour, weight="bold", va="top", ha="right")
        self.y -= self._line_h(9.5)

    def banner(self, text: str, *, colour: str) -> None:
        """A boxed note. Used for the disclaimer and for a crisis pathway."""
        size = 8.5
        lines = []
        for block in str(text).split("\n"):
            lines.extend(textwrap.wrap(block, self._chars(size, 0.22)) or [""])
        height = len(lines) * self._line_h(size) + 0.22 / _PAGE_H_IN
        self._room(height + 0.08)
        from matplotlib.patches import FancyBboxPatch

        self.fig.add_artist(FancyBboxPatch(
            (self._left, self.y - height), self._width, height,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            linewidth=0.9, edgecolor=colour, facecolor=colour + "14",
            transform=self.fig.transFigure, zorder=0,
        ))
        self.y -= 0.11 / _PAGE_H_IN
        for line in lines:
            self.fig.text(self._left + 0.11 / _PAGE_W_IN, self.y, line,
                          size=size, color=colour, va="top", ha="left")
            self.y -= self._line_h(size)
        self.y -= 0.11 / _PAGE_H_IN
        self.space(0.10)


def _level_colour(level: str) -> str:
    return _ALERT if str(level).lower() not in ("normal", "not measured",
                                                NOT_MEASURED.lower()) else _INK


def render_pdf(summary: Mapping[str, Any]) -> bytes:
    """
    The summary as PDF bytes.

    matplotlib is imported here rather than at module import, because the app
    reaches this module on startup to get the two names and only reaches the
    renderer when a check-in has finished. Importing it eagerly put about half
    a second on every cold start for a file most sessions never ask for.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    title = "Auracare check-in summary"
    sheet = _Sheet(title)

    # -- who and when -----------------------------------------------------
    sheet.para(title, size=17, weight="bold")
    sheet.space(0.04)
    when = str(summary.get("generated_at") or "").replace("T", " at ")
    who = summary.get("patient_id")
    sheet.para(
        f"Patient {who} · {when}" if who else f"Completed {when}",
        size=9, colour=_MUTED,
    )
    sheet.space(0.10)
    sheet.rule()

    # -- the caveat, before any number ------------------------------------
    sheet.banner(str(summary.get("disclaimer") or DISCLAIMER), colour=_MUTED)

    # -- a crisis pathway goes above everything else ----------------------
    if summary.get("crisis") and summary.get("crisis_resources"):
        sheet.heading("If you need help now")
        sheet.banner(str(summary["crisis_resources"]), colour=_ALERT)

    # -- the questionnaire -------------------------------------------------
    phq8 = dict(summary.get("phq8") or {})
    sheet.heading("How you have been feeling")
    if phq8.get("scorable") and phq8.get("score") is not None:
        sheet.row(
            "PHQ-8 total",
            f"{phq8['score']} of {phq8.get('max', MAX_SCORE)}"
            + (f" · {phq8['band']}" if phq8.get("band") else ""),
        )
        if phq8.get("prorated"):
            sheet.para(
                "Some questions were not answered, so the total is scaled "
                "from the ones that were. Treat it as approximate.",
                size=8.5, colour=_MUTED,
            )
        sheet.para(
            "The PHQ-8 is a questionnaire about the last two weeks, scored "
            "against published cut-points. It is a measure of what you "
            "reported, not a diagnosis of anything.",
            size=8.5, colour=_MUTED,
        )
    else:
        answered = phq8.get("n_answered")
        sheet.para(
            "Not scored"
            + (f" – {answered} of 8 questions answered." if answered
               is not None else "."),
            colour=_MUTED,
        )

    # -- the speaking tasks ------------------------------------------------
    tasks = list(summary.get("tasks") or [])
    sheet.heading("What was measured")
    if tasks:
        for task in tasks:
            level = task.get("level") or NOT_MEASURED
            sheet.row(
                f"Task {task.get('task_id')} · {task.get('title', '')}",
                str(level), colour=_level_colour(level),
            )
        measured = summary.get("n_tasks_measured")
        if measured is not None:
            sheet.space(0.04)
            sheet.para(f"{measured} of {len(tasks)} tasks produced a "
                       "measurement.", size=8.5, colour=_MUTED)
    else:
        sheet.para("No speaking task produced a measurement.", colour=_MUTED)

    # -- Task 1's three numbers, the ones she asked for by name ------------
    voice = {k: _f(v) for k, v in (summary.get("voice") or {}).items()}
    if any(v is not None for v in voice.values()):
        sheet.heading("Your voice during the held 'ah'")
        for key, (label, unit) in _VOICE_LABELS.items():
            value = voice.get(key)
            sheet.row(label, "not measured" if value is None
                      else f"{value:.2f} {unit}".strip(),
                      colour=_MUTED if value is None else _INK)
        sheet.space(0.04)
        sheet.para(
            "These describe how steady and how clear the sound was. They are "
            "engineering measures taken from one short recording, and a "
            "single reading is affected by a cold, a dry throat, tiredness "
            "or the microphone itself.",
            size=8.5, colour=_MUTED,
        )

    # -- what happens next -------------------------------------------------
    messages = [m for m in (summary.get("patient_messages") or []) if m]
    pathway = summary.get("pathway_label")
    if messages or pathway:
        sheet.heading("What happens next")
        if pathway:
            sheet.row("Suggested next step", str(pathway),
                      colour=_ALERT if summary.get("crisis") else _OK)
            sheet.space(0.04)
        for message in messages:
            sheet.bullet(str(message))

    # -- footer on every page ---------------------------------------------
    for number, fig in enumerate(sheet.pages, start=1):
        fig.text(
            _MARGIN_IN / _PAGE_W_IN, _MARGIN_IN / _PAGE_H_IN * 0.55,
            "Auracare screening prototype · simulated throughout · "
            "not a medical device",
            size=7.5, color=_MUTED, va="center", ha="left",
        )
        fig.text(
            1.0 - _MARGIN_IN / _PAGE_W_IN, _MARGIN_IN / _PAGE_H_IN * 0.55,
            f"Page {number} of {len(sheet.pages)}",
            size=7.5, color=_MUTED, va="center", ha="right",
        )

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in sheet.pages:
            pdf.savefig(fig, facecolor="white")
        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "Auracare screening prototype"
        info["Subject"] = "Screening summary - not a diagnosis"
        # Pinned to the summary's own timestamp rather than now, so rendering
        # the same summary twice gives the same bytes. The tests rest on that.
        try:
            info["CreationDate"] = dt.datetime.fromisoformat(
                str(summary.get("generated_at"))
            )
        except (TypeError, ValueError):
            pass
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Offering it
# ---------------------------------------------------------------------------
def get_pdf_download_component(
    summary: Mapping[str, Any],
    label: str = "Download your summary (PDF)",
) -> str:
    """

    """
    try:
        data = base64.b64encode(render_pdf(summary)).decode("ascii")
    except Exception:  # noqa: BLE001 - a missing takeaway must not end a run
        return (
            '<p style="font:13px system-ui;color:#b3261e;margin:0">'
            "The summary could not be prepared for download.</p>"
        )
    name = _html.escape(summary_filename(summary), quote=True)
    return (
        '<a href="data:application/pdf;base64,' + data + '" '
        f'download="{name}" target="_blank" rel="noopener" '
        'style="display:inline-block;font:600 14px/1.2 system-ui,sans-serif;'
        'background:#ff4b4b;color:#fff;padding:10px 18px;border-radius:8px;'
        'text-decoration:none">' + _html.escape(label) + "</a>"
    )


# ---------------------------------------------------------------------------
# Emailing it
# ---------------------------------------------------------------------------
_MAILTO_BUDGET = 1400


def summary_subject(summary: Mapping[str, Any]) -> str:
    who = summary.get("patient_id")
    when = str(summary.get("generated_at") or "")[:10]
    return (
        f"Auracare check-in summary - {who} - {when}" if who
        else f"Auracare check-in summary - {when}"
    )


def summary_as_text(summary: Mapping[str, Any]) -> str:
    """
    The same document as plain text, for the body of an email.
    """
    lines: List[str] = []
    who = summary.get("patient_id")
    when = str(summary.get("generated_at") or "").replace("T", " at ")
    lines.append(f"Auracare check-in summary{f' for {who}' if who else ''}")
    lines.append(f"Completed {when}")
    lines.append("")

    if summary.get("crisis") and summary.get("crisis_resources"):
        lines.append("IF YOU NEED HELP NOW")
        lines.append(str(summary["crisis_resources"]))
        lines.append("")

    phq8 = dict(summary.get("phq8") or {})
    if phq8.get("scorable") and phq8.get("score") is not None:
        band = f" ({phq8['band']})" if phq8.get("band") else ""
        lines.append(
            f"PHQ-8: {phq8['score']} of {phq8.get('max', MAX_SCORE)}{band}"
        )
    else:
        lines.append("PHQ-8: not scored")

    measured = summary.get("n_tasks_measured")
    tasks = list(summary.get("tasks") or [])
    if tasks:
        lines.append(f"Speaking tasks measured: {measured} of {len(tasks)}")
    if summary.get("pathway_label"):
        lines.append(f"Suggested next step: {summary['pathway_label']}")
    lines.append("")

    for message in (summary.get("patient_messages") or []):
        text = f"- {message}"
        if sum(len(x) + 1 for x in lines) + len(text) > _MAILTO_BUDGET:
            lines.append("- (more detail in the attached PDF)")
            break
        lines.append(text)

    lines.append("")
    lines.append(
        "This is a screening prototype, not a medical device and not a "
        "diagnosis. The full one-page summary is the attached PDF; download "
        "it from the app and attach it to this message."
    )
    return "\n".join(lines)


def mailto_link(summary: Mapping[str, Any], to: str = "") -> str:
    """
    A `mailto:` the patient's own mail client opens, already written.

    Deliberately not a send. The app states on every screen and on the PDF
    that nothing is transmitted anywhere, and standing up an outbound mail
    path would make that false - the check-in would start posting health data
    off the machine, from a prototype, with no consent step and nowhere
    audited to send it. Handing the draft to the patient's own client keeps
    the sending where it belongs, with the person whose results these are,
    and works offline and inside a hospital network.
    """
    from urllib.parse import quote

    body = summary_as_text(summary)
    return (
        f"mailto:{quote(str(to or '').strip())}"
        f"?subject={quote(summary_subject(summary))}"
        f"&body={quote(body)}"
    )
