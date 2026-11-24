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

@dataclass(frozen=True)
class TaskScript:

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

    animation: Optional[str] = None

TASK_SCRIPTS: List[TaskScript] = [
    TaskScript(
        task_id=1,
        domain="Respiratory & laryngeal health",
        window="0:00 - 0:25",
        record_seconds=10,

        prompt=(
            "For our first task, take a deep, comfortable breath. When the "
            "timer starts, say the 'ah' sound and hold it steadily for as "
            "long as you comfortably can until the bar fills up."
        ),
        instruction="Inhale as the circle grows, then hold **\u201cAhhh\u201d**.",

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

        prompt=(
            "Next up is Task 2. For this task, we're going to evaluate your "
            "exhalation dynamics using your microphone. Please take a deep "
            "breath in, fill your lungs completely, and when you're ready, "
            "blow out as hard and fast as you can into the microphone until "
            "your lungs are empty, just like you're blowing out a big "
            "birthday candle."
        ),
        instruction="Deep breath in\u2026 then **blow out hard and fast**.",
        cue_class="candles-exhale",
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

WELCOME_SCRIPT = (
    "Hi there, I'm Annie from Auracare. Let's take two quick minutes for your "
    "voice and respiratory check-in today. Find a comfortable spot and "
    "whenever you're ready, let's begin."
)

COMPLETION_PROMPT = (
    "Check-in complete. Your results have been added to your health record. "
    "Have a good day."
)

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

  .cut-points { margin-top: 6px; }
  .cut-row {
      display: flex; justify-content: space-between; align-items: baseline;
      gap: 12px; padding: 3px 0; font-size: 0.88rem;

      color: inherit; opacity: 0.85;
  }
  .cut-row span:last-child {
      font-variant-numeric: tabular-nums; font-weight: 700; opacity: 1;
  }
  .cue-box {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 26px 18px; background: #0f172a;
      border-radius: 16px; border: 1px solid #334155; text-align: center;

      aspect-ratio: 354 / 402;
      min-height: 260px;
  }
  .cue-title { margin-top: 12px; margin-bottom: 4px; font-weight: 600; }

  .st-key-annie-voice [data-testid="stAudio"] {
      position: absolute;
      width: 1px; height: 1px;
      opacity: 0;
      overflow: hidden;
      pointer-events: none;
  }

  .st-key-annie-voice.auracare-needs-tap [data-testid="stAudio"] {
      position: static;

      display: block;
      width: 100%;

      height: auto; min-height: 54px;
      opacity: 1;
      pointer-events: auto;
  }
  .st-key-annie-voice.auracare-needs-tap::before {
      content: "Press play to hear Annie.";
      display: block;
      color: #94a3b8; font-size: 0.86rem; margin-bottom: 4px;
  }

  .st-key-annie-voice-quiet [data-testid="stAudio"] {
      position: absolute;
      width: 1px; height: 1px;
      opacity: 0;
      overflow: hidden;
      pointer-events: none;
  }

  .st-key-annie-spoke,
  .st-key-quiet-spoke {
      visibility: hidden;
      position: absolute;
      height: 0;
      overflow: hidden;
  }

  .st-key-task-cue {
      background: #000;
      border: 1px solid #334155;
      border-radius: 16px;
      aspect-ratio: 354 / 402;
      min-height: 260px;
      overflow: hidden;
      padding: 10px;
  }
  .st-key-task-cue [data-testid="stVerticalBlock"] {
      height: 100%;
      justify-content: center;
  }
  .st-key-task-cue img, .st-key-task-cue video {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      margin: 0 auto;
      display: block;
  }
  .cue-hint { color: #94a3b8; font-size: 0.86rem; margin: 0; }

  .st-key-task-footer [data-testid="stButton"] button {
      background: #fff !important;
      color: #0f172a !important;
      border: 1px solid #cbd5e1 !important;
      font-weight: 700;
      min-height: 44px;
  }
  .st-key-task-footer [data-testid="stButton"] button:hover {
      background: #f1f5f9 !important;
      color: #0f172a !important;
      border-color: #94a3b8 !important;
  }

  .circle-phonation, .circle-ddk {

      width: 160px; height: 160px; border-radius: 50%;

      margin: 32px 0;
  }
  .circle-phonation {
      background: radial-gradient(circle, #38bdf8 0%, #0284c7 100%);
      animation: pulse-chest 4s infinite ease-in-out;
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
  @keyframes rhythm-ddk {
      0%   { transform: scale(0.9); }
      100% { transform: scale(1.12); }
  }

  .candles-exhale {

      width: 160px; height: 160px; margin: 32px 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
  }
  .cake-flames {
      display: flex; align-items: flex-end; justify-content: center; gap: 12px;
  }
  .candle {
      position: relative; width: 7px; height: 34px; border-radius: 3px 3px 0 0;
      background: repeating-linear-gradient(
          180deg, #fde68a 0 6px, #fb923c 6px 12px);
  }
  .flame {
      position: absolute; left: 50%; bottom: 34px; margin-left: -5.5px;
      width: 11px; height: 15px;
      border-radius: 50% 50% 50% 50% / 62% 62% 38% 38%;
      background: radial-gradient(circle at 50% 70%,
          #fff7ed 0%, #fbbf24 45%, #f97316 100%);
      box-shadow: 0 0 14px 3px rgba(249,115,22,0.75);
      transform-origin: 50% 100%;
      animation: blow-out 4s infinite ease-in-out;
  }

  .candle:nth-child(2) .flame { animation-delay: 0.12s; }
  .candle:nth-child(3) .flame { animation-delay: 0.24s; }
  .candle:nth-child(4) .flame { animation-delay: 0.36s; }
  .candle:nth-child(5) .flame { animation-delay: 0.48s; }
  .cake-icing {
      width: 118px; height: 15px; border-radius: 8px 8px 3px 3px;
      background: linear-gradient(180deg, #fda4af 0%, #fb7185 100%);
  }
  .cake-base {
      width: 118px; height: 44px; border-radius: 0 0 10px 10px;
      background: linear-gradient(180deg, #c2854f 0%, #8b5e34 100%);
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.28);
  }
  @keyframes blow-out {
      0%, 14%  { transform: scale(1) skewX(0deg); opacity: 1; }
      20%      { transform: scale(1.08) skewX(-14deg); opacity: 1; }
      28%      { transform: scale(0.72) skewX(-26deg); opacity: 0.85; }
      36%, 80% { transform: scale(0) skewX(-30deg); opacity: 0; }
      90%      { transform: scale(1.06) skewX(0deg); opacity: 1; }
      100%     { transform: scale(1) skewX(0deg); opacity: 1; }
  }
  @media (prefers-reduced-motion: reduce) {
      .circle-phonation, .circle-ddk, .flame { animation: none; }
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

VOICE_CONTAINER_KEY = "annie-voice"
VOICE_CONTAINER_CLASS = "st-key-annie-voice"

QUIET_CONTAINER_KEY = "annie-voice-quiet"
QUIET_CONTAINER_CLASS = "st-key-annie-voice-quiet"

SPOKE_CONTAINER_KEY = "annie-spoke"
SPOKE_CONTAINER_CLASS = "st-key-annie-spoke"

QUIET_SPOKE_CONTAINER_KEY = "quiet-spoke"
QUIET_SPOKE_CONTAINER_CLASS = "st-key-quiet-spoke"

VOICE_NEEDS_TAP_CLASS = "auracare-needs-tap"

CUE_CONTAINER_KEY = "task-cue"
CUE_CONTAINER_CLASS = "st-key-task-cue"

CUE_BODIES: Dict[str, str] = {
    "candles-exhale": (
        '<div class="candles-exhale">'
        '<div class="cake-flames">'
        + '<span class="candle"><span class="flame"></span></span>' * 5
        + "</div>"
        '<div class="cake-icing"></div>'
        '<div class="cake-base"></div>'
        "</div>"
    ),
}

def cue_html(script: TaskScript) -> str:

    if not script.cue_class:
        return ""
    body = CUE_BODIES.get(
        script.cue_class, f'<div class="{script.cue_class}"></div>'
    )
    return f"""
    <div class="cue-box">
      {body}
      <div class="cue-title" style="color:{script.colour}">{script.cue_label}</div>
      <p class="cue-hint">{script.cue_hint}</p>
    </div>
    """

def speak_script(text: str, nonce: str, rate: float = 0.95) -> str:

    payload = json.dumps(text)
    return f"""
    <script>
      (function () {{
        const key = "auracare-spoken-{nonce}";
        try {{
          if (window.sessionStorage.getItem(key)) return;
          window.sessionStorage.setItem(key, "1");
        }} catch (e) {{  }}
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

def advance_when_clip_ends(nonce: str, button_text: str) -> str:

    return """
    <script>
      (function () {
        const clip = __CLIP__;
        const wanted = __LABEL__;
        const doc = window.parent ? window.parent.document : document;

        const press = function () {
          const dialog = doc.querySelector('[role="dialog"]');
          const root = dialog || doc;
          const buttons = root.querySelectorAll("button");
          for (const b of buttons) {
            if ((b.innerText || "").trim() === wanted) {
              if (b.dataset.auracareAdvanced === clip) return true;
              b.dataset.auracareAdvanced = clip;
              b.click();
              return true;
            }
          }
          return false;
        };

        const watch = function (m) {
          if (m.dataset.auracareWatched === clip) return true;
          m.dataset.auracareWatched = clip;
          if (m.ended) { press(); return true; }
          m.addEventListener("ended", press, {once: true});
          m.addEventListener("pause", function () {
            if (!m.ended && m.currentTime > 0.3) press();
          });
          return true;
        };

        let attempts = 240;
        let waited = 0;
        const tick = function () {
          const media = doc.querySelectorAll("video, audio");
          for (const m of media) {
            if (m.closest(".st-key-task-cue")) continue;
            if (m.dataset.auracareNudged !== clip) continue;
            watch(m);
            return;
          }
          waited += 1;
          if (waited > 20) {
            for (const m of doc.querySelectorAll("video[autoplay], audio[autoplay]")) {
              if (m.closest(".st-key-task-cue")) continue;
              if (m.dataset.auracareWatched) continue;
              watch(m);
              return;
            }
          }
          if (--attempts > 0) window.setTimeout(tick, 500);
        };
        tick();
      })();
    </script>
    """.replace("__CLIP__", json.dumps(nonce)).replace(
        "__LABEL__", json.dumps(button_text)
    )

def press_when_clip_plays(nonce: str, container_class: str) -> str:

    return """
    <script>
      (function () {
        const clip = __CLIP__;
        const sel = __SEL__;
        const doc = window.parent ? window.parent.document : document;
        const press = function () {
          const b = doc.querySelector(sel + " button");
          if (!b) return false;
          if (b.dataset.auracareSpoke === clip) return true;
          b.dataset.auracareSpoke = clip;
          b.click();
          return true;
        };
        let attempts = 600;
        const tick = function () {
          const media = doc.querySelectorAll("video, audio");
          for (const m of media) {
            if (m.dataset.auracareNudged !== clip) continue;
            if (m.closest(".st-key-task-cue")) continue;
            if (m.closest(".st-key-annie-voice-quiet")) continue;
            if (m.currentTime > 0.2 || m.ended) { press(); return; }
          }
          if (--attempts > 0) window.setTimeout(tick, 500);
        };
        tick();
      })();
    </script>
    """.replace("__CLIP__", json.dumps(str(nonce))).replace(
        "__SEL__", json.dumps("." + str(container_class))
    )

def keep_videos_inline() -> str:

    return """
    <script>
      (function () {
        const win = window.parent || window;
        const doc = win.document;
        const inline = function (m) {
          if (!m || m.tagName !== "VIDEO") return;
          if (m.dataset.auracareInline === "1") return;
          m.dataset.auracareInline = "1";
          m.playsInline = true;
          m.setAttribute("playsinline", "");
          m.setAttribute("webkit-playsinline", "");
        };
        const sweep = function (root) {
          if (!root || !root.querySelectorAll) return;
          if (root.tagName === "VIDEO") inline(root);
          root.querySelectorAll("video").forEach(inline);
        };
        sweep(doc);
        try {
          if (win.__auracareInlineObserver) {
            win.__auracareInlineObserver.disconnect();
          }
        } catch (e) {  }
        const obs = new MutationObserver(function (records) {
          for (const r of records) {
            r.addedNodes.forEach(sweep);
          }
        });
        obs.observe(doc.body, {childList: true, subtree: true});
        win.__auracareInlineObserver = obs;
      })();
    </script>
    """

def press_when_quiet_clip_plays(
    nonce: str, container_class: str, fallback_ms: int = 3000
) -> str:

    return """
    <script>
      (function () {
        const clip = __CLIP__;
        const sel = __SEL__;
        const doc = window.parent ? window.parent.document : document;
        const press = function () {
          const b = doc.querySelector(sel + " button");
          if (!b) return false;
          if (b.dataset.auracareQuietSpoke === clip) return true;
          b.dataset.auracareQuietSpoke = clip;
          b.click();
          return true;
        };
        let attempts = 140;
        const tick = function () {
          const media = doc.querySelectorAll("audio, video");
          for (const m of media) {
            if (m.dataset.auracareNudged !== clip) continue;
            if (m.currentTime > 0.05 || m.ended) { press(); return; }
          }
          if (--attempts > 0) window.setTimeout(tick, 150);
        };
        tick();
        window.setTimeout(press, __FALLBACK__);
      })();
    </script>
    """.replace("__CLIP__", json.dumps(str(nonce))).replace(
        "__SEL__", json.dumps("." + str(container_class))
    ).replace("__FALLBACK__", str(int(fallback_ms)))

def autoplay_nudge(nonce: str) -> str:

    return """
    <script>
      (function () {
        const clip = __CLIP__;
        const doc = window.parent ? window.parent.document : document;

        const ownClip = function () {
          const frame = window.frameElement;
          if (!frame) return null;
          let node = frame.closest('[data-testid="stElementContainer"]')
                     || frame;
          for (let up = 0; up < 3 && node && node !== doc.body; up++) {
            for (let s = node.previousElementSibling; s;
                 s = s.previousElementSibling) {
              const m = s.matches("video, audio")
                        ? s : s.querySelector("video, audio");
              if (m && !m.closest(".st-key-task-cue")) return m;
            }
            node = node.parentElement;
          }
          return null;
        };

        const start = function (m) {
            if (m.closest(".st-key-task-cue")) return;
            const src = m.currentSrc || m.src || "";
            if (!src) return;
            if (m.dataset.auracareNudged === clip) return;
            m.dataset.auracareNudged = clip;
            if (m.tagName === "VIDEO") {
              m.playsInline = true;
              m.setAttribute("playsinline", "");
              m.setAttribute("webkit-playsinline", "");
            }
            if (m.currentTime > 0.05 || m.ended) {
              try { m.currentTime = 0; } catch (e) {  }
            }
            const started = m.play();
            if (started && started.catch) {
              started.catch(function () {
                m.dataset.auracareBlocked = "1";
                const quiet = m.closest(".st-key-annie-voice-quiet");
                const holder = m.closest(".st-key-annie-voice") || quiet;
                if (!holder) return;
                const retry = function () {
                  const again = m.play();
                  if (again && again.then) {
                    again.then(function () {
                      holder.classList.remove("auracare-needs-tap");
                    }).catch(function () {});
                  }
                };
                doc.addEventListener("pointerdown", retry, {once: true});
                doc.addEventListener("keydown", retry, {once: true});
                if (quiet) return;
                window.setTimeout(function () {
                  if (m.paused && m.currentTime === 0) {
                    holder.classList.add("auracare-needs-tap");
                  }
                }, 1200);
              });
            }
        };

        let attempts = 24;
        const tick = function () {
          const own = ownClip();
          if (own) {
            start(own);
          } else {
            doc.querySelectorAll(
              "video[autoplay], audio[autoplay], .st-key-annie-voice audio,"
              + " .st-key-annie-voice-quiet audio"
            ).forEach(function (m) {
              if (!m.dataset.auracareNudged) start(m);
            });
          }
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

    if value is None:
        return "not measured"
    text = f"{value:.{nd}f}"
    return f"{text} {unit}".strip()

__all__ = [
    "COMPLETION_PROMPT",
    "CRISIS_PROMPT",
    "CSS",
    "CUE_BODIES",
    "CUE_CONTAINER_CLASS",
    "CUE_CONTAINER_KEY",
    "QUIET_CONTAINER_CLASS",
    "QUIET_CONTAINER_KEY",
    "QUIET_SPOKE_CONTAINER_CLASS",
    "QUIET_SPOKE_CONTAINER_KEY",
    "SPOKE_CONTAINER_CLASS",
    "SPOKE_CONTAINER_KEY",
    "press_when_clip_plays",
    "keep_videos_inline",
    "press_when_quiet_clip_plays",
    "VOICE_CONTAINER_CLASS",
    "VOICE_CONTAINER_KEY",
    "VOICE_NEEDS_TAP_CLASS",
    "EMERGENCY_PROMPT",
    "TASK_SCRIPTS",
    "TaskScript",
    "advance_when_clip_ends",
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
