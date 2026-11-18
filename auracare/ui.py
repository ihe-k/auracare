"""
Presentation helpers for the Streamlit app: CSS, animated task cues, spoken
prompts and the protocol script.

Kept out of `app.py` so the protocol copy - the words the agent actually says -
sits in one reviewable place rather than scattered through UI code.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .risk import (
    BORDERLINE,
    CRITICAL,
    ELEVATED,
    LEVEL_COLOUR,
    NORMAL,
    NOT_MEASURED,
)


# ---------------------------------------------------------------------------
# Protocol script
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskScript:
    """One step of the guided check-in."""

    task_id: int
    domain: str
    window: str
    record_seconds: int
    prompt: str
    instruction: str
    cue_class: str
    cue_label: str
    cue_hint: str
    colour: str
    safety_check: str
    # Filename of the clinician's Annie animation for this task, inside
    # assets/animations/. None means no clip exists for the task, which
    # degrades to the written prompt rather than borrowing another task's
    # clip; all four tasks now have their own.
    animation: Optional[str] = None


TASK_SCRIPTS: List[TaskScript] = [
    TaskScript(
        task_id=1,
        domain="Respiratory & laryngeal health",
        window="0:00 - 0:25",
        record_seconds=10,
        # Clinician's wording, and the text her recorded Annie audio speaks.
        # The welcome line she wrote sits on the check-in screen separately;
        # this is her task 1 script.
        prompt=(
            "For our first task, take a deep, comfortable breath. When the "
            "timer starts, say the 'ah' sound and hold it steadily for as "
            "long as you comfortably can until the bar fills up."
        ),
        instruction="Inhale as the circle grows, then hold **\u201cAhhh\u201d**.",
        # The title read "Sustained phonation /a:/". That is the IPA for
        # the vowel: correct notation for a phonetician, meaningless to a
        # patient, so it came off the patient-facing title at the
        # clinician's request. The domain name on the clinical side still
        # says what is being measured.
        cue_class="circle-phonation",
        cue_label="Task 1 \u00b7 Sustained phonation",
        cue_hint="Hold the note steadily until you run out of breath.",
        animation="task1.mp4",
        colour="#38bdf8",
        safety_check=(
            "Sustained phonation time and vocal perturbation. Phonation below "
            "the critical threshold raises the respiratory intercept."
        ),
    ),
    TaskScript(
        task_id=2,
        domain="Expiratory airway dynamics",
        window="0:25 - 0:45",
        record_seconds=10,
        # Wording supplied by the clinician, 10 Nov. Kept close to her text:
        # the instruction is what shapes the manoeuvre, and a cleaner, more
        # consistent manoeuvre is the one real lever on Task 2 accuracy.
        prompt=(
            "Next up is Task 2. For this task, we're going to evaluate your "
            "exhalation dynamics using your microphone. Please take a deep "
            "breath in, fill your lungs completely, and when you're ready, "
            "blow out as hard and fast as you can into the microphone until "
            "your lungs are empty, just like you're blowing out a big "
            "birthday candle."
        ),
        instruction="Deep breath in\u2026 then **blow out hard and fast**.",
        cue_class="circle-exhale",
        cue_label="Task 2 \u00b7 Forced exhalation",
        cue_hint="One sharp, complete breath out. Keep going until empty.",
        animation="task2.mp4",
        colour="#f97316",
        safety_check=(
            "Exhalation duration, rise time and the tonal-peak prominence used "
            "as a wheeze proxy."
        ),
    ),
    TaskScript(
        task_id=3,
        domain="Neuromotor speech control",
        window="0:45 - 1:05",
        record_seconds=10,
        prompt=(
            "Next up, let's test your speech agility. When you're ready, "
            "repeat the sequence 'pa-ta-ka' clearly and steadily over and "
            "over at a comfortable rhythm until I let you know."
        ),
        instruction="Match the rhythm: **\u201cpa - ta - ka - pa - ta - ka\u201d**",
        cue_class="circle-ddk",
        cue_label="Task 3 \u00b7 Diadochokinetic rate",
        cue_hint="As fast as you can while staying clear and even.",
        animation="task3.mp4",
        colour="#c084fc",
        safety_check=(
            "Syllable repetition rate and the variability of inter-syllable "
            "timing."
        ),
    ),
    TaskScript(
        task_id=4,
        domain="Cognitive & language fluency",
        window="1:05 - 1:45",
        record_seconds=30,
        prompt=(
            "For our final step, please look at the picture on your screen. "
            "Take a moment to look it over, and then tell me everything you "
            "see happening in the scene, and how you think the people are "
            "feeling."
        ),
        instruction="Describe **everything you can see happening** in the picture.",
        cue_class="",
        cue_label="Task 4 \u00b7 Picture description",
        cue_hint="Keep going for the full thirty seconds if you can.",
        animation="task4.mp4",
        colour="#34d399",
        safety_check=(
            "Speech pause ratio, speech rate, lexical diversity and the "
            "negative-language lexicon."
        ),
    ),
]

WELCOME_HEADING = "Welcome to your check-in"

# Transcribed from `intro.mp4` so the written version is what Annie actually
# says. It is on a screen of its own, which is why it can state how long the
# whole thing takes: on Task 1 that line arrived after the check-in had
# already started.
WELCOME_SCRIPT = (
    "Hi there, I'm Annie from Auracare. Let's take two quick minutes for your "
    "voice and respiratory check-in today. Find a comfortable spot and "
    "whenever you're ready, let's begin."
)

COMPLETION_PROMPT = (
    "Check-in complete. Your results have been added to your health record. "
    "Have a good day."
)

# The clinician's own opening line for the closing screen. It goes on the two
# non-escalating pathways only. Spoken before an emergency message it would
# have said "I'm so happy you completed this screening" immediately before
# "your breathing sounds strained, we are routing your assessment to
# emergency triage", which is the wrong tone at the worst possible moment.
SCREENING_THANKS = "I'm so happy you completed this screening."

CLINICIAN_REVIEW_PROMPT = (
    "Check-in complete. We would like a clinician to look at your results, "
    "and someone will be in touch."
)

EMERGENCY_PROMPT = (
    "Based on your check-in, your breathing sounds strained. We are routing "
    "your assessment to emergency triage now. Please stay on the line. If you "
    "are in severe distress, call 999 straight away."
)

CRISIS_PROMPT = (
    "Thank you for telling me that. Some of what you said suggests you may be "
    "having thoughts of harming yourself. Please talk to someone today. "
    "Samaritans are free on 116 123, any time."
)


def script_for(task_id: int) -> TaskScript:
    for script in TASK_SCRIPTS:
        if script.task_id == task_id:
            return script
    raise KeyError(f"no script for task {task_id}")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
  .aura-agent {
      background: #0f172a; padding: 14px 18px; border-radius: 12px;
      margin-bottom: 10px; border-left: 4px solid #0284c7; color: #f8fafc;
      font-size: 0.97rem; line-height: 1.5;
  }
  .aura-user {
      background: #334155; padding: 12px 18px; border-radius: 12px;
      margin-bottom: 10px; border-right: 4px solid #10b981; color: #f8fafc;
      text-align: right; font-size: 0.93rem;
  }
  .aura-system {
      background: #1e293b; padding: 10px 16px; border-radius: 10px;
      margin-bottom: 10px; border-left: 4px solid #64748b; color: #cbd5e1;
      font-size: 0.86rem; font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  /* The published PHQ-8 cut-points in the sidebar. Space-between pins every
     number to the same right edge, which padded text cannot do in a
     proportional font. */
  .cut-points { margin-top: 6px; }
  .cut-row {
      display: flex; justify-content: space-between; align-items: baseline;
      gap: 12px; padding: 3px 0; font-size: 0.88rem;
      /* Inherited, not fixed. The slate greys used elsewhere in this sheet
         are chosen against the dark panels they sit on; in the sidebar under
         a light theme they came out almost invisible, which `st.text` never
         did because it follows the theme. */
      color: inherit; opacity: 0.85;
  }
  .cut-row span:last-child {
      font-variant-numeric: tabular-nums; font-weight: 700; opacity: 1;
  }
  .cue-box {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 26px 18px; background: #0f172a;
      border-radius: 16px; border: 1px solid #334155; text-align: center;
      /* The task clips are 354x402. Giving the panel the same shape makes
         the two halves of the screen identical in size at any window width,
         which a fixed height cannot do: the video's height follows its
         column. The floor is for very narrow windows, where the columns
         stack and the ratio would otherwise squash the text. */
      aspect-ratio: 354 / 402;
      min-height: 260px;
  }
  .cue-title { margin-top: 12px; margin-bottom: 4px; font-weight: 600; }
  .cue-hint { color: #94a3b8; font-size: 0.86rem; margin: 0; }

  .circle-phonation, .circle-exhale, .circle-ddk {
      /* Scaled up with the panel. At 118px it was a dot in the middle of a
         lot of empty navy once the panel grew to match the video. */
      width: 160px; height: 160px; border-radius: 50%;
      /* `transform: scale()` paints outside the element without reserving
         any room for it, so the circle grew over the sentence beneath it -
         what the clinician reported on 17 Nov. The widest animation is the
         exhale burst at 1.3, which puts 24px past each edge
         ((160 x 1.3) - 160) / 2. Reserving exactly that much stops the
         overlap but still leaves the sentence only 10px clear of the circle
         at its widest, which reads as crowded. 32px here plus the title's
         own 12px leaves a measured 20px of clear space at the tightest
         instant of the widest animation. */
      margin: 32px 0;
  }
  .circle-phonation {
      background: radial-gradient(circle, #38bdf8 0%, #0284c7 100%);
      animation: pulse-chest 4s infinite ease-in-out;
  }
  .circle-exhale {
      background: radial-gradient(circle, #f97316 0%, #ea580c 100%);
      animation: burst-exhale 2s infinite ease-in-out;
  }
  .circle-ddk {
      background: radial-gradient(circle, #a855f7 0%, #9333ea 100%);
      animation: rhythm-ddk 0.2s infinite alternate ease-in-out;
  }
  @keyframes pulse-chest {
      0%   { transform: scale(0.85); box-shadow: 0 0 10px rgba(56,189,248,0.2); }
      50%  { transform: scale(1.25); box-shadow: 0 0 35px rgba(56,189,248,0.7); }
      100% { transform: scale(0.85); box-shadow: 0 0 10px rgba(56,189,248,0.2); }
  }
  @keyframes burst-exhale {
      0%   { transform: scale(1.3); opacity: 1; }
      70%  { transform: scale(0.65); opacity: 0.5; }
      100% { transform: scale(1.3); opacity: 1; }
  }
  @keyframes rhythm-ddk {
      0%   { transform: scale(0.9); }
      100% { transform: scale(1.12); }
  }
  @media (prefers-reduced-motion: reduce) {
      .circle-phonation, .circle-exhale, .circle-ddk { animation: none; }
  }

  .pill {
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 0.76rem; font-weight: 700; color: #0b1120;
      letter-spacing: 0.02em;
  }
  .sim-banner {
      background: #422006; border: 1px solid #a16207; color: #fde68a;
      padding: 10px 14px; border-radius: 10px; font-size: 0.85rem;
      margin-bottom: 12px;
  }
  .step-dot {
      display: inline-block; width: 11px; height: 11px; border-radius: 50%;
      margin-right: 7px; border: 2px solid #475569;
  }
</style>
"""


def level_pill(level: str) -> str:
    colour = LEVEL_COLOUR.get(level, "#64748b")
    return f'<span class="pill" style="background:{colour}">{html.escape(level)}</span>'


def agent_bubble(text: str) -> str:
    return (
        # The agent is Annie, the clinician's named avatar - her scripts say
        # "I'm Annie from Auracare" and the badge on her artwork reads
        # "Annie, triage nurse". Auracare is the service, not the speaker.
        '<div class="aura-agent"><strong>Annie</strong><br>'
        f"{html.escape(text)}</div>"
    )


def patient_bubble(text: str) -> str:
    return (
        '<div class="aura-user"><strong>You</strong><br>'
        f"{html.escape(text)}</div>"
    )


def system_bubble(text: str) -> str:
    return f'<div class="aura-system">{html.escape(text)}</div>'


def cue_html(script: TaskScript) -> str:
    """The animated visual pacing guide for one task."""
    if not script.cue_class:
        return ""
    return f"""
    <div class="cue-box">
      <div class="{script.cue_class}"></div>
      <div class="cue-title" style="color:{script.colour}">{script.cue_label}</div>
      <p class="cue-hint">{script.cue_hint}</p>
    </div>
    """


def speak_script(text: str, nonce: str, rate: float = 0.95) -> str:
    """
    Speak `text` with the browser's own speech synthesis.

    Real text-to-speech with no API key and no network call - the voice is the
    one already installed on the patient's device. `nonce` must change whenever
    a new utterance should be spoken, because Streamlit re-renders components
    on every interaction and the utterance would otherwise repeat.

    Browsers block speech until the page has had a user gesture, so the first
    prompt may stay silent until the patient clicks something. That is a
    browser policy, not a bug, and the text is always shown as well.
    """
    payload = json.dumps(text)
    return f"""
    <script>
      (function () {{
        const key = "auracare-spoken-{nonce}";
        try {{
          if (window.sessionStorage.getItem(key)) return;
          window.sessionStorage.setItem(key, "1");
        }} catch (e) {{ /* private mode: speak anyway */ }}
        const synth = window.speechSynthesis;
        if (!synth) return;
        synth.cancel();
        const u = new SpeechSynthesisUtterance({payload});
        u.rate = {rate};
        u.pitch = 1.0;
        u.lang = "en-GB";
        synth.speak(u);
      }})();
    </script>
    """


def autoplay_nudge(nonce: str) -> str:
    """
    Ask Annie's freshly rendered clip to start playing.

    Chrome, Edge and Safari all ignore the `autoplay` attribute on an element
    that was inserted after the page loaded, which is every clip in this app:
    Streamlit swaps the media in over its websocket on each rerun, long after
    the document finished loading. The attribute is there, the element is
    there, and the browser simply never starts it. Measured in Chromium with
    `--autoplay-policy=user-gesture-required`, which is the real default: the
    task prompt, the affirmation and the quiet-retry clip all sat at
    `paused: true, currentTime: 0`.

    Calling `play()` explicitly is permitted, and was verified to resolve
    rather than reject once the patient has clicked anything at all - the
    Begin button is enough. So the clips are nudged by hand instead.

    One guard matters, and `nonce` is it. Each element records the nonce it
    was last nudged for, so a rerun that renders the same nonce leaves a
    playing clip alone, while a new nonce rewinds and restarts. The caller is
    what decides which is which: `app.clip_nonce` combines the filename with
    an epoch that moves on every phase change, so Annie repeats herself when
    the conversation moves on and stays quiet when the page merely redraws.

    `nonce` must identify the clip, for the same reason `speak_script` takes
    one. This is rendered through `components.html`, which Streamlit serves as
    an iframe carrying the markup in `srcdoc`. Identical markup at the same
    position is reused on a rerun and the script inside it never runs a second
    time, so a constant string nudged the task prompt and then sat inert while
    the affirmation and the quiet-retry clip rendered. Putting the clip's name
    in the markup changes the `srcdoc`, which forces a fresh iframe and a
    fresh run.

    Nothing here can make the very first clip play, because the welcome video
    renders before the patient has clicked anything and the browser has no
    gesture to work with. That one keeps its visible play control.
    """
    return """
    <script>
      (function () {
        const clip = __CLIP__;  // forces a distinct iframe per clip
        const doc = window.parent ? window.parent.document : document;
        let attempts = 24;
        const tick = function () {
          const media = doc.querySelectorAll("video[autoplay], audio[autoplay]");
          media.forEach(function (m) {
            const src = m.currentSrc || m.src || "";
            if (!src) return;
            // Keyed on this rendering, not on the file. Answering "Say that
            // again" on a task with no clarification clip replays the prompt
            // clip, the same file that has just finished; keyed on the file
            // it looked already-played and stayed silent.
            if (m.dataset.auracareNudged === clip) return;
            m.dataset.auracareNudged = clip;
            if (m.currentTime > 0.05 || m.ended) {
              try { m.currentTime = 0; } catch (e) { /* not seekable yet */ }
            }
            const started = m.play();
            if (started && started.catch) {
              started.catch(function () { m.dataset.auracareBlocked = "1"; });
            }
          });
          // Keep looking for the whole window rather than stopping at the
          // first element found. A phase change renders this script and swaps
          // the clip's source in separate frames, and the earlier version
          // gave up on the first pass because the previous clip - already
          // nudged, and still carrying the old source - was sitting there.
          // The affirmation never played, which is exactly what the clinician
          // reported. The per-element guard above is what stops a replay, so
          // polling on is free.
          if (--attempts > 0) { setTimeout(tick, 250); }
        };
        tick();
      })();
    </script>
    """.replace("__CLIP__", json.dumps(str(nonce)))


def stop_speech() -> str:
    return (
        "<script>if (window.speechSynthesis) window.speechSynthesis.cancel();"
        "</script>"
    )


def progress_dots(current: int, total: int, done: Dict[int, bool]) -> str:
    out = []
    for i in range(1, total + 1):
        if done.get(i):
            colour = "#10b981"
        elif i == current:
            colour = "#38bdf8"
        else:
            colour = "transparent"
        out.append(
            f'<span class="step-dot" style="background:{colour}"></span>'
        )
    return "".join(out)


def fmt_value(value: Optional[float], unit: str = "", nd: int = 2) -> str:
    """Render a measurement, or an explicit dash when it could not be taken."""
    if value is None:
        return "not measured"
    text = f"{value:.{nd}f}"
    return f"{text} {unit}".strip()


__all__ = [
    "COMPLETION_PROMPT",
    "CRISIS_PROMPT",
    "CSS",
    "EMERGENCY_PROMPT",
    "TASK_SCRIPTS",
    "TaskScript",
    "agent_bubble",
    "cue_html",
    "fmt_value",
    "level_pill",
    "patient_bubble",
    "progress_dots",
    "script_for",
    "autoplay_nudge",
    "speak_script",
    "stop_speech",
    "system_bubble",
]
