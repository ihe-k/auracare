"""
The patient's takeaway PDF.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from auracare import pdf_export
from auracare.config import PROVISIONAL_DEFAULTS, Thresholds
from auracare.phq8 import MAX_SCORE, PHQ8_ITEMS, score_phq8
from auracare.risk import CRISIS_RESOURCES_UK, assess_screening

WHEN = dt.datetime(2026, 11, 19, 21, 40)


def _responses(scores):
    return {item.key: s for item, s in zip(PHQ8_ITEMS, scores)}


def _assessment(scores, *, task1=None):
    return assess_screening(
        task1, None, None, None, score_phq8(_responses(scores)),
        Thresholds(dict(PROVISIONAL_DEFAULTS)),
    )


def _summary(scores=(2, 3, 2, 1, 2, 3, 1, 2), **kw):
    kw.setdefault("assessment", _assessment(scores))
    kw.setdefault("generated_at", WHEN)
    return pdf_export.generate_diagnostic_summary(_responses(scores), **kw)


# ---------------------------------------------------------------------------
# Her call shape
# ---------------------------------------------------------------------------
def test_her_two_argument_call_still_works():
    """
    Her `app.py` calls this with the PHQ-8 answers and Task 1's three numbers
    and nothing else. The extra arguments this build passes are keyword-only
    so that call keeps working rather than becoming a TypeError at the end of
    a check-in.
    """
    summary = pdf_export.generate_diagnostic_summary(
        {"phq_1": 2}, {"jitter": 0.8, "shimmer": 2.5, "hnr": 16.0}
    )
    assert summary["voice"]["jitter_local_pct"] == pytest.approx(0.8)
    assert summary["voice"]["hnr_db"] == pytest.approx(16.0)


def test_both_spellings_of_the_voice_metrics_are_accepted():
    """Her call site says `jitter`; the feature module says `jitter_local_pct`."""
    short = pdf_export.generate_diagnostic_summary(None, {"shimmer": 3.1})
    long = pdf_export.generate_diagnostic_summary(None, {"shimmer_local_pct": 3.1})
    assert short["voice"] == long["voice"]


def test_the_module_exports_the_two_names_her_code_imports():
    for name in ("generate_diagnostic_summary", "get_pdf_download_component"):
        assert hasattr(pdf_export, name)


# ---------------------------------------------------------------------------
# What the document must say
# ---------------------------------------------------------------------------
def test_the_disclaimer_is_carried_on_the_page():
    """
    A screenshot of a result is explained by the screen around it. A PDF is
    not: it gets printed, forwarded and read later by someone who never saw
    the app, so it has to carry its own caveats.
    """
    summary = _summary()
    for phrase in ("not a medical device", "not a diagnosis", "provisional"):
        assert phrase in summary["disclaimer"]


def test_a_severe_score_puts_the_helplines_on_the_page():
    """
    The crisis *pathway* only fires on a self-harm disclosure in the Task 4
    transcript, but `assess_screening` adds the helplines to the patient
    messages from band 3 up. Wherever the app has decided the patient should
    see that number, this page gives it a box of its own rather than leaving
    it as the fourth bullet of a closing section.
    """
    summary = _summary((3, 3, 3, 3, 3, 3, 3, 3))
    assert summary["crisis"] is True
    assert summary["crisis_pathway"] is False, "PHQ-8 alone is not the pathway"
    assert summary["crisis_resources"] == CRISIS_RESOURCES_UK
    assert "116 123" in summary["crisis_resources"]


def test_the_helplines_are_printed_once_not_twice():
    """Promoted into its own box, so it leaves the bullet list."""
    summary = _summary((3, 3, 3, 3, 3, 3, 3, 3))
    assert CRISIS_RESOURCES_UK not in summary["patient_messages"]


def test_a_routine_run_does_not_print_crisis_resources():
    summary = _summary((0, 0, 0, 0, 0, 0, 0, 0))
    assert summary["crisis"] is False
    assert summary["crisis_resources"] is None


def test_the_questionnaire_is_scored_by_the_assessment_not_again_here():
    """
    The assessment already applies the proration rule for a partly answered
    form. Re-scoring from the raw answers would be a second implementation of
    the same rule and the two would drift.
    """
    scores = (2, 3, 2, 1, 2, 3, 1, 2)
    summary = _summary(scores)
    assert summary["phq8"]["score"] == score_phq8(_responses(scores)).total_score
    assert summary["phq8"]["max"] == MAX_SCORE
    assert summary["phq8"]["band"]


# ---------------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------------
def test_an_unmeasured_task_one_drops_its_voice_numbers():
    """
    The three numbers come from Task 1. A page that lists Task 1 as not
    measured and then prints its jitter contradicts itself in front of the
    patient. Her draft reached the same place from the other side, with
    `getattr(r1.voice, "jitter_local_pct", 0.8)` putting a plausible invented
    figure on a document a patient keeps.
    """
    summary = _summary(assessment=_assessment((1,) * 8))
    assert all(v is None for v in summary["voice"].values()), summary["voice"]


def test_a_missing_measurement_stays_missing():
    summary = pdf_export.generate_diagnostic_summary(
        None, {"jitter": None, "shimmer": float("nan"), "hnr": "n/a"}
    )
    assert summary["voice"] == {
        "jitter_local_pct": None, "shimmer_local_pct": None, "hnr_db": None
    }


def test_nothing_in_the_payload_raises_on_junk():
    """
    A summary that cannot be built is worse than one with gaps: the gaps are
    visible on the page, a traceback at the end of a check-in loses the run.
    """
    for bad in (None, "not a mapping", 17, [1, 2, 3]):
        summary = pdf_export.generate_diagnostic_summary(bad, bad, assessment=bad)
        assert isinstance(summary, dict)
        assert pdf_export.render_pdf(summary).startswith(b"%PDF")


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------
def test_it_is_a_pdf():
    assert pdf_export.render_pdf(_summary()).startswith(b"%PDF")


def test_the_same_summary_renders_the_same_bytes():
    """
    The creation date is pinned to the summary's own timestamp rather than
    now, so a patient who downloads twice gets the same document.
    """
    summary = _summary()
    assert pdf_export.render_pdf(summary) == pdf_export.render_pdf(summary)


def test_the_filename_says_whose_it_is_and_sorts_by_date():
    name = pdf_export.summary_filename(_summary(patient_id="NHS-948-204-11"))
    assert name == "auracare-summary-NHS-948-204-11-2026-11-19.pdf"


def test_a_filename_cannot_escape_its_folder():
    """The patient ID is a free-text sidebar field."""
    name = pdf_export.summary_filename(
        _summary(patient_id="../../etc/passwd")
    )
    assert "/" not in name and "\\" not in name and ".." not in name


def test_the_download_component_carries_the_file_with_it():
    """
    Base64'd into the link rather than fetched, so the component has no
    server to call and works on a machine with no internet.
    """
    html = pdf_export.get_pdf_download_component(_summary())
    assert "data:application/pdf;base64," in html
    assert "download=" in html
    assert "http://" not in html and "https://" not in html


def test_the_download_component_never_raises():
    html = pdf_export.get_pdf_download_component({"generated_at": object()})
    assert isinstance(html, str) and html


# ---------------------------------------------------------------------------
# Wired into the app
# ---------------------------------------------------------------------------
def _app_source() -> str:
    return Path(__file__).resolve().parents[1].joinpath("app.py").read_text(
        encoding="utf-8"
    )


def test_the_summary_is_built_once_when_the_check_in_ends():
    """
    Not re-derived per tab. The sidebar thresholds are tunable while the app
    is open, so deriving it on every rerun would let a threshold nudged
    afterwards rewrite a summary already downloaded under the same filename.
    """
    source = _app_source()
    assert "def build_diagnostic_summary()" in source
    build = source.index("build_diagnostic_summary()\n    st.rerun()")
    finished = source.index("st.session_state.finished = True")
    assert finished < build


def test_the_app_offers_the_download_everywhere_she_asked():
    """Completion screen, the patient's Results tab, and the clinical view."""
    source = _app_source()
    assert source.count("offer_summary_download(") == 4   # 1 def + 3 calls
    for key in ("summary_done", "summary_results", "summary_clinical"):
        assert f'"{key}"' in source


def test_starting_over_clears_the_previous_takeaway():
    source = _app_source()
    reset = source[source.index("def reset_all()"):]
    reset = reset[: reset.index("init_state()")]
    assert '"diagnostic_summary"' in reset


def test_a_message_that_points_below_is_re_aimed_at_this_layout():
    """
    `assess_screening` ends its severe-band message with "reach out to one of
    the services below", true of the Results tab where the helplines follow.
    Here they are boxed near the top, so the sentence would send the reader
    down the page past the thing it means.
    """
    summary = _summary((3, 3, 3, 3, 3, 3, 3, 3))
    joined = " ".join(summary["patient_messages"])
    assert "services below" not in joined
    assert "top of this page" in joined


def test_a_routine_run_leaves_its_messages_alone():
    summary = _summary((0, 0, 0, 0, 0, 0, 0, 0))
    expected = _assessment((0,) * 8).patient_messages
    assert summary["patient_messages"] == [m for m in expected if m]


def test_the_document_is_drawn_once_not_once_per_tab():
    """
    Streamlit runs the body of every tab on every rerun, so rendering at each
    of the three call sites drew the same document three times a keystroke
    for as long as the finished screen stayed open.
    """
    source = _app_source()
    build = source[source.index("def build_diagnostic_summary()"):]
    build = build[: build.index("def offer_summary_download(")]
    assert "render_pdf(summary)" in build

    offer = source[source.index("def offer_summary_download("):]
    offer = offer[: offer.index("def check_intercept(")]
    assert "render_pdf(" not in offer, "the call site must serve cached bytes"
    assert 'st.session_state.get("diagnostic_pdf")' in offer


def test_starting_over_clears_the_rendered_bytes_too():
    source = _app_source()
    reset = source[source.index("def reset_all()"):]
    reset = reset[: reset.index("init_state()")]
    assert '"diagnostic_pdf"' in reset


# ---------------------------------------------------------------------------
# Emailing it
# ---------------------------------------------------------------------------
def test_the_email_is_a_draft_not_a_send():
    """
    The app says on every screen and on the PDF that nothing is transmitted
    anywhere. An outbound mail path would make that false, so this hands the
    draft to the patient's own client instead.
    """
    link = pdf_export.mailto_link(_summary(), "her@example.nhs.uk")
    assert link.startswith("mailto:her%40example.nhs.uk?")
    assert "subject=" in link and "body=" in link


def test_an_empty_recipient_still_opens_a_draft():
    link = pdf_export.mailto_link(_summary())
    assert link.startswith("mailto:?")


def test_the_email_body_fits_what_a_mail_client_will_take():
    """
    Outlook is the strict one, at about 2000 characters for the whole URL.
    The detail stays in the attachment rather than being silently truncated.
    """
    link = pdf_export.mailto_link(
        _summary((3, 3, 3, 3, 3, 3, 3, 3)), "someone@example.com"
    )
    assert len(link) < 2000, len(link)


def test_the_email_carries_the_headline_and_the_caveat():
    text = pdf_export.summary_as_text(_summary())
    assert "PHQ-8" in text
    assert "not a medical device" in text
    assert "attach" in text.lower(), "it must say the PDF is not attached"


def test_a_severe_score_puts_the_helplines_at_the_top_of_the_email():
    text = pdf_export.summary_as_text(_summary((3, 3, 3, 3, 3, 3, 3, 3)))
    assert "IF YOU NEED HELP NOW" in text
    assert text.index("116 123") < text.index("PHQ-8")


def test_the_subject_says_whose_and_when():
    subject = pdf_export.summary_subject(_summary(patient_id="NHS-948-204-11"))
    assert "NHS-948-204-11" in subject and "2026-11-19" in subject


def test_the_app_offers_the_email_beside_every_download():
    source = _app_source()
    assert "def offer_summary_email(" in source
    offer = source[source.index("def offer_summary_download("):]
    offer = offer[: offer.index("def offer_summary_email(")]
    assert "offer_summary_email(key)" in offer, (
        "the email sits with the download, so all three places get both"
    )


def test_the_app_does_not_send_mail_itself():
    """No SMTP, no outbound post. The draft goes to the patient's client."""
    source = _app_source()
    for forbidden in ("smtplib", "sendmail", "send_message", "requests.post"):
        assert forbidden not in source, forbidden
