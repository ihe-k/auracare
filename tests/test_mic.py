"""
The microphone diagnosis.
"""

import pytest

from auracare import mic


@pytest.mark.parametrize(
    "url",
    [
        "https://auracare.streamlit.app/",
        "https://example.com:8501/",
        "http://localhost:8501/",
        "http://localhost/",
        "http://127.0.0.1:8501/",
        "http://[::1]:8501/",
        "http://box.localhost:8501/",
    ],
)
def test_pages_the_browser_trusts_are_not_reported_as_blocked(url):
    assert mic.diagnose(url).blocked is False


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.5:8501/",
        "http://10.0.0.12:8501/",
        "http://172.16.4.9:8501/",
        "http://my-laptop.local:8501/",
    ],
)
def test_plain_http_on_anything_but_localhost_is_blocked(url):
    context = mic.diagnose(url)
    assert context.blocked is True
    assert context.reason
    assert context.suggested_url


def test_the_fix_offered_is_the_same_page_on_localhost():
    context = mic.diagnose("http://192.168.1.5:8501/")
    assert context.suggested_url == "http://localhost:8501/"


def test_a_non_default_port_is_carried_into_the_suggestion():
    context = mic.diagnose("http://192.168.1.5:9123/")
    assert context.suggested_url == "http://localhost:9123/"


def test_no_url_means_no_claim():
    """
    """
    for value in (None, ""):
        context = mic.diagnose(value)
        assert context.blocked is False
        assert context.reason == ""


def test_a_secure_context_is_never_reported_as_working():
    """
    The rule is one-sided on purpose. https means "not blocked by this rule",
    not "the microphone works" - permission can still be denied and the
    machine may have no input device.
    """
    context = mic.diagnose("https://example.com/")
    assert context.secure is True
    assert context.blocked is False
    # No claim is made beyond that.
    assert context.reason == ""


def test_the_other_causes_are_listed_in_a_useful_order():
    titles = [title for title, _ in mic.TROUBLESHOOTING]
    assert len(titles) >= 4
    # The address is first because it is the only cause the app can be
    # certain of, and the one that matches what she reported.
    assert "localhost" in titles[0]
    for _, detail in mic.TROUBLESHOOTING:
        assert len(detail) > 40
