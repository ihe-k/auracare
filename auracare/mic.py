"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

# Hosts a browser treats as trustworthy over plain http. Firefox, Chrome and
# Safari all follow the W3C secure-contexts rule here; ".localhost" is in the
# specification too and Chrome honours it.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", ""})


@dataclass(frozen=True)
class MicContext:
    """What the browser's address implies about microphone access."""

    url: Optional[str]
    secure: bool
    reason: str = ""
    suggested_url: Optional[str] = None

    @property
    def blocked(self) -> bool:
        """True only when we are certain the browser will refuse the mic."""
        return self.url is not None and not self.secure


def _is_local(host: str) -> bool:
    host = host.lower()
    if host in LOCAL_HOSTS:
        return True
    # "mybox.localhost" and friends.
    return host.endswith(".localhost")


def localhost_equivalent(url: str) -> Optional[str]:
    """The same page on localhost, which is the fix in almost every case."""
    parts = urlsplit(url)
    if not parts.hostname:
        return None
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit(("http", f"localhost{port}", parts.path or "/", "", ""))


def diagnose(url: Optional[str]) -> MicContext:
    """
    Decide whether the browser will allow the microphone on this page.

    Deliberately one-sided. A secure context is necessary for the microphone
    but not sufficient - permission can still be denied, and the machine may
    have no input device - so `secure` true means "not blocked by this rule",
    never "the microphone works". Only the insecure case is reported as
    certain, because that one is certain.
    """
    if not url:
        # Older Streamlit, or a context where the URL is not exposed. Say
        # nothing rather than warn about a situation we cannot see.
        return MicContext(url=None, secure=True)

    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    host = parts.hostname or ""

    if scheme == "https":
        return MicContext(url=url, secure=True)

    if _is_local(host):
        return MicContext(url=url, secure=True)

    if scheme in ("http", ""):
        return MicContext(
            url=url,
            secure=False,
            reason=(
                f"This page is open at {host}, over http. Browsers only allow "
                "microphone access on https or on localhost, so the record "
                "button cannot start - it fails silently, with no error."
            ),
            suggested_url=localhost_equivalent(url),
        )

    return MicContext(url=url, secure=True)


TROUBLESHOOTING = [
    (
        "Open the app at localhost",
        "When Streamlit starts it prints a Local URL and a Network URL. Use "
        "the Local URL (http://localhost:8501). The Network URL, the one with "
        "numbers like 192.168.x.x, cannot use the microphone on any browser.",
    ),
    (
        "Allow the microphone for this site",
        "Click the padlock or the camera icon in the address bar and set "
        "Microphone to Allow, then reload. If you pressed Block earlier, the "
        "browser remembers it and will not ask again.",
    ),
    (
        "Let the browser reach the microphone (Windows)",
        "Settings, then Privacy & security, then Microphone. Turn on "
        "Microphone access and Let apps access your microphone, and check "
        "your browser is listed and enabled.",
    ),
    (
        "Check the microphone itself",
        "Settings, then System, then Sound. Pick the input device and watch "
        "the level move as you speak. A headset that is connected but not "
        "selected is the most common cause.",
    ),
    (
        "Try a different browser",
        "Chrome, Edge and Firefox all work. Safari on iPhone needs the page "
        "to be https. If the prototype is running on your own machine, "
        "Chrome or Edge on that machine is the simplest route.",
    ),
]

__all__ = [
    "LOCAL_HOSTS",
    "MicContext",
    "TROUBLESHOOTING",
    "diagnose",
    "localhost_equivalent",
]
