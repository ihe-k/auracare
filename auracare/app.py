"""
Auracare: Guided Vocal Biomarkers Screening - prototype.

Run with:  streamlit run app.py

Scope and honesty, stated once here and repeated in the UI:
  * This is a screening prototype, not a medical device and not a diagnosis.
  * Acoustic thresholds are provisional engineering values, tunable from the
    sidebar, and separate from the published clinical reference values.
  * Depression severity is scored from the PHQ-8 against published cut-points.
  * All emergency dispatches and FHIR messages are simulated. Nothing is
    transmitted anywhere.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from auracare import flow, mic, paths, ui
from auracare.imaging import alzheimer
from auracare.audio_io import AudioDecodeError, check_quality, load_audio
from auracare.config import (
    CLINICAL_REFERENCE,
    PROVISIONAL_DEFAULTS,
    SignalParams,
    Thresholds,
)
from auracare.features.ddk import analyse_ddk
from auracare.features.expiratory import analyse_expiratory
from auracare.features.fluency import analyse_fluency
from auracare.features.phonation import analyse_phonation
from auracare.features.voice import PARSELMOUTH_AVAILABLE
from auracare.fhir import SIMULATION_BANNER, build_bundle
from auracare.phq8 import (
    INTRO_HEADING,
    INTRO_SCRIPT,
    MAX_SCORE,
    PHQ8_ITEMS,
    RECALL_STEM_SINGLE,
    RESPONSE_OPTIONS,
    empty_responses,
    score_phq8,
)
from auracare.risk import (
    CRISIS_RESOURCES_UK,
    ELEVATED,
    LEVEL_RANK,
    NOT_MEASURED,
    PATHWAY_CRISIS,
    PATHWAY_EMERGENCY,
    PATHWAY_LABEL,
    PATHWAY_ROUTINE,
    assess_screening,
)
from auracare.stt import analyse_language, available_backends, transcribe

st.set_page_config(
    page_title="Auracare: Guided Vocal Biomarkers Screening",
    page_icon="\U0001fa7a",
    layout="wide",
)

ANALYSERS = {
    1: analyse_phonation,
    2: analyse_expiratory,
    3: analyse_ddk,
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "step": 1,
        "recordings": {},       # task_id -> raw bytes
        "fingerprints": {},     # task_id -> sha1 of the bytes analysed
        "results": {},          # task_id -> analysis result object
        "quality": {},          # task_id -> QualityReport
        "transcript": None,
        "phq8_responses": empty_responses(),
        "thresholds": Thresholds(),
        "patient_id": "NHS-948-204-11",
        "log": [],
        "intercepts": [],
        "finished": False,
        "spoken": set(),
        # Tasks the patient could not complete, so the report can say "not
        # attempted" rather than silently omitting them.
        "skipped": set(),
        # -- conversation state ------------------------------------------
        # Which phase the current task is in, and which task that phase
        # belongs to. The second half matters: a phase stored on its own
        # follows the patient into the next task.
        "phase": flow.PHASE_PROMPT,
        "phase_step": 1,
        # Bumped whenever Annie is meant to speak again. It separates a
        # deliberate repeat, which must restart the clip, from an incidental
        # rerun, which must not. See `play_clip`.
        "clip_epoch": 0,
        # task_id -> (fingerprint, reason). Recordings judged unusable, kept
        # by fingerprint so the same file cannot be judged twice. Without it
        # the retry paths loop forever: st.audio_input hands back the same
        # bytes on every rerun.
        "rejected": {},
        # task_id -> monotonic deadline for the quiet prompt.
        "silence_deadline": {},
        # task_id -> how many times the instruction has been repeated, so the
        # browser voice speaks again rather than deduplicating itself.
        "clarify_count": {},
        # -- PHQ-8 flow ---------------------------------------------------
        # Her script creates these lazily on first use, which means "Start
        # over" leaves the patient on question 6 of the previous run.
        "phq_intro_done": False,
        "phq_index": 0,
        # Annie's greeting is a screen of its own, ahead of Task 1.
        "welcome_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_agent(text: str) -> None:
    if not st.session_state.log or st.session_state.log[-1] != ("agent", text):
        st.session_state.log.append(("agent", text))


def log_patient(text: str) -> None:
    st.session_state.log.append(("patient", text))


def log_system(text: str) -> None:
    st.session_state.log.append(("system", text))


def reset_all() -> None:
    for key in (
        "step", "recordings", "fingerprints", "results", "quality",
        "transcript", "phq8_responses", "log", "intercepts", "finished",
        "spoken", "skipped", "phase", "phase_step", "rejected",
        "silence_deadline", "clarify_count", "phq_intro_done", "phq_index",
        "welcome_done",
    ):
        st.session_state.pop(key, None)
    init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Dict[str, object]:
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.text_input("Patient ID", key="patient_id")

    st.sidebar.markdown("#### Speech-to-text")
    backends = available_backends()
    if backends:
        stt_on = st.sidebar.toggle(
            "Transcribe Task 4", value=True,
            help="Needed for negative-language flagging and word-rate measures.",
        )
        model_size = st.sidebar.selectbox(
            "Whisper model",
            ["tiny.en", "base.en", "small.en"],
            index=1,
            help="tiny is roughly 3x faster; base is more accurate. "
                 "Expect transcription to take about as long as the recording.",
        )
    else:
        stt_on = False
        model_size = "base.en"
        st.sidebar.warning(
            "No speech-to-text backend installed. Acoustic measures still "
            "work; transcript-based measures will be skipped.\n\n"
            "`pip install faster-whisper`"
        )

    speak = st.sidebar.toggle(
        "Speak prompts aloud", value=False,
        help="Uses your browser's built-in speech synthesis. No audio leaves "
             "this device.",
    )

    st.sidebar.markdown("#### Conversation")
    watch_for_silence = st.sidebar.toggle(
        "Notice when it goes quiet", value=True, key="watch_for_silence",
        help="If nothing is recorded within a few seconds of Annie finishing "
             "the instruction, she asks again instead of waiting silently. "
             "The wait is the length of her clip plus "
             f"{flow.SILENCE_GRACE_SECONDS:.0f} seconds.",
    )

    st.sidebar.markdown("#### Provisional thresholds")
    st.sidebar.caption(
        "Prototype operating points. Tunable, and deliberately separate from "
        "the published clinical values below."
    )
    thresholds: Thresholds = st.session_state.thresholds
    group_labels = {
        "task1": "Task 1 · phonation",
        "task2": "Task 2 · exhalation",
        "task3": "Task 3 · DDK",
        "task4": "Task 4 · fluency",
    }
    for group, label in group_labels.items():
        with st.sidebar.expander(label):
            for key, default in PROVISIONAL_DEFAULTS[group].items():
                current = thresholds.get(group, key)
                step = 0.01 if abs(default) < 2 else 0.1
                new = st.number_input(
                    key.replace("_", " "),
                    value=float(current),
                    step=float(step),
                    format="%.3f",
                    key=f"thr_{group}_{key}",
                )
                if new != current:
                    thresholds.set(group, key, new)

    with st.sidebar.expander("PHQ-8 cut-points (published)"):
        st.caption(
            "These are the Kroenke 2009 cut-points. They are shown for "
            "transparency and are not intended to be tuned."
        )
        # Laid out rather than padded. `st.text` renders in the app's
        # proportional font, so the `:<24` padding that was here lined the
        # numbers up in the source and left them ragged on screen, which is
        # what the clinician photographed on 17 Nov. A flex row pins every
        # number to the same right edge at any sidebar width.
        rows = "".join(
            '<div class="cut-row"><span>{}</span><span>{}</span></div>'.format(
                html.escape(key.replace("_", " ")), html.escape(str(value))
            )
            for key, value in PROVISIONAL_DEFAULTS["phq8"].items()
        )
        st.markdown(f'<div class="cut-points">{rows}</div>',
                    unsafe_allow_html=True)

    # Full width, not half. The second column was never used, and at half
    # the sidebar's width the label broke as "Reset threshold / s".
    if st.sidebar.button("Reset thresholds", use_container_width=True):
        thresholds.reset()
        st.rerun()
    st.sidebar.download_button(
        "Export JSON",
        data=thresholds.to_json(),
        file_name="auracare_thresholds.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded = st.sidebar.file_uploader(
        "Import thresholds", type=["json"], key="thr_upload"
    )
    if uploaded is not None:
        try:
            st.session_state.thresholds = Thresholds.from_dict(
                json.loads(uploaded.getvalue().decode("utf-8"))
            )
            st.sidebar.success("Thresholds loaded.")
        except Exception as exc:
            st.sidebar.error(f"Could not read that file: {exc}")

    st.sidebar.markdown("#### Environment")
    found = paths.describe()
    st.sidebar.caption(
        f"Parselmouth: {'ready' if PARSELMOUTH_AVAILABLE else 'MISSING'}  \n"
        f"Speech-to-text: {', '.join(backends) or 'none'}  \n"
        f"Severity model: {'found' if found['severity_model'] else 'not found'}"
    )

    if st.sidebar.button("\U0001f501 Start over", type="secondary",
                         use_container_width=True):
        reset_all()
        st.rerun()

    return {
        "stt_on": stt_on,
        "model_size": model_size,
        "speak": speak,
        "watch_for_silence": watch_for_silence,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
# Distinguishes "no transcript was supplied, work it out" from "a transcript
# was supplied and it is None". Task 4's help scan has to transcribe before
# deciding whether to measure at all, and transcription is the slowest step in
# the app; passing the result through avoids doing it twice.
_UNSET = object()


def analyse_task(
    task_id: int,
    raw: bytes,
    settings: Dict[str, object],
    transcript=_UNSET,
) -> None:
    """Decode, quality-check and analyse one task recording."""
    fingerprint = hashlib.sha1(raw).hexdigest()
    if st.session_state.fingerprints.get(task_id) == fingerprint:
        return  # already analysed this exact recording

    params = SignalParams()
    try:
        audio = load_audio(raw)
    except AudioDecodeError as exc:
        st.error(f"Could not read that recording: {exc}")
        return

    quality = check_quality(audio)
    st.session_state.quality[task_id] = quality

    with st.spinner(f"Analysing Task {task_id}…"):
        if task_id == 4:
            if transcript is _UNSET:
                transcript = None
                if settings["stt_on"]:
                    transcript = transcribe(
                        audio, model_size=str(settings["model_size"])
                    )
                st.session_state.transcript = transcript
            result = analyse_fluency(audio, params, transcript=transcript)
        else:
            result = ANALYSERS[task_id](audio, params)

    st.session_state.results[task_id] = result
    st.session_state.fingerprints[task_id] = fingerprint

    log_patient(f"(recorded Task {task_id}, {audio.duration_s:.1f} s)")
    log_system(summarise_result(task_id, result))


def summarise_result(task_id: int, result) -> str:
    """One compact telemetry line for the conversation log."""
    if task_id == 1:
        return (
            f"telemetry · phonation {ui.fmt_value(result.phonation_duration_s, 's')}"
            f" · jitter {ui.fmt_value(result.voice.jitter_local_pct, '%')}"
            f" · shimmer {ui.fmt_value(result.voice.shimmer_local_pct, '%')}"
            f" · HNR {ui.fmt_value(result.voice.hnr_db, 'dB')}"
        )
    if task_id == 2:
        return (
            f"telemetry · exhalation {ui.fmt_value(result.exhalation_duration_s, 's')}"
            f" · rise {ui.fmt_value(result.rise_time_s, 's')}"
            f" · wheeze index {ui.fmt_value(result.wheeze_index_db, 'dB')}"
        )
    if task_id == 3:
        return (
            f"telemetry · rate {ui.fmt_value(result.syllable_rate_hz, 'syll/s')}"
            f" · {result.n_syllables} syllables"
            f" · rhythm CV {ui.fmt_value(result.rhythm_cv, '', 3)}"
        )
    # The two hesitation measures are shown alongside the pause ratio because
    # they are what decides whether a high ratio escalates. Showing the ratio
    # on its own would leave the clinician unable to see why it did or did not.
    return (
        f"telemetry · pause ratio {ui.fmt_value(result.pause_ratio, '', 3)}"
        f" · speech rate {ui.fmt_value(result.speech_rate_sylls, 'syll/s')}"
        f" · {result.n_pauses} pauses"
        f" · retrieval latency "
        f"{ui.fmt_value(result.word_retrieval_latency_s, 's')}"
        f" · mean unfilled pause "
        f"{ui.fmt_value(result.mean_unfilled_pause_s, 's')}"
        f" · longest unfilled "
        f"{ui.fmt_value(result.longest_unfilled_pause_s, 's')}"
    )


def current_assessment():
    """Grade whatever has been measured so far."""
    return assess_screening(
        st.session_state.results.get(1),
        st.session_state.results.get(2),
        st.session_state.results.get(3),
        st.session_state.results.get(4),
        score_phq8(st.session_state.phq8_responses),
        st.session_state.thresholds,
    )


def check_intercept(task_id: int) -> Optional[str]:
    """
    Did this task trip a safety intercept?

    Only fires on a *measured* breach. A recording that could not be measured
    yields None from the extractor and is never an intercept - the original
    draft showed its emergency banner before any analysis had run at all.
    """
    assessment = current_assessment()
    task = assessment.task(task_id)
    if task is not None and task.intercept_triggered:
        return task.intercept_reason
    return None


# ---------------------------------------------------------------------------
# Conversation phase
# ---------------------------------------------------------------------------
def current_phase(step: int) -> str:
    """
    The phase this task step is in, always a member of `flow.PHASES`.

    Resets whenever the step changes, so a CLARIFY entered on Task 2 cannot
    follow the patient into Task 3. A bare `phase` string with no record of
    which task it belongs to, which is how the clinician's sketch holds it,
    leaks across every advance.
    """
    if st.session_state.get("phase_step") != step:
        st.session_state.phase_step = step
        st.session_state.phase = flow.PHASE_PROMPT
        st.session_state.silence_deadline.pop(step, None)
    phase = str(st.session_state.get("phase") or flow.PHASE_PROMPT)
    return phase if phase in flow.PHASES else flow.PHASE_PROMPT


def clip_nonce(clip) -> str:
    """Identify one rendering of one clip, for the autoplay nudge."""
    return f"{clip.name}#{st.session_state.get('clip_epoch', 0)}"


def bump_clip_epoch() -> None:
    """Mark that Annie should speak again, even if it is the same clip."""
    st.session_state.clip_epoch = st.session_state.get("clip_epoch", 0) + 1


def set_phase(step: int, phase: str) -> None:
    """Move this task to `phase`. Returning to PROMPT restarts the quiet wait."""
    st.session_state.phase = phase
    st.session_state.phase_step = step
    # A phase change is Annie taking her turn, so whatever clip it renders is
    # meant to be heard - even when it is the same file that just played. See
    # `play_clip` for why that needs saying out loud.
    bump_clip_epoch()
    if phase == flow.PHASE_PROMPT:
        st.session_state.silence_deadline.pop(step, None)


def rejection_reason(step: int) -> Optional[str]:
    """Why the last recording for this task was not measured, if it was not."""
    entry = st.session_state.rejected.get(step)
    return entry[1] if entry else None


# ---------------------------------------------------------------------------
# Annie's clips
# ---------------------------------------------------------------------------
AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg", ".opus")


def play_clip(filename: Optional[str], *, autoplay: bool = True) -> bool:
    """
    Render one of Annie's clips, video or audio. False if it is not installed.

    Returning a boolean rather than raising is what lets every phase degrade
    to text: a task with no clarification clip repeats the instruction in
    writing instead of reaching a dead end.

    The nudge is keyed on the clip *and* the current epoch rather than on the
    filename alone. Tasks without their own clarification clip answer "Say
    that again" by replaying the prompt clip, which is the same file that has
    just finished playing. Keyed on the filename, the nudge treated that as a
    clip it had already started and left it silent - the screen changed and
    Annie said nothing, which is what the clinician reported on 17 Nov. The
    epoch moves on every phase change, so a deliberate repeat restarts and an
    incidental rerun still does not.
    """
    clip = paths.find_animation(filename)
    if clip is None:
        return False
    if clip.suffix.lower() in AUDIO_SUFFIXES:
        fmt = "audio/mpeg" if clip.suffix.lower() == ".mp3" else "audio/wav"
        st.audio(str(clip), format=fmt, autoplay=autoplay)
        if autoplay:
            components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)
        return True
    # The clip fills its column, so it matches the cue panel beside it. It
    # used to be held inside a further st.columns([2, 3]) because the files
    # are only 354x402 and upscale poorly, but shrinking one side to protect
    # the resolution is what left the screen lopsided. The column is now half
    # the content width rather than three fifths, so the upscale is modest.
    st.video(str(clip), autoplay=autoplay)
    if autoplay:
        components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)
    return True


# ---------------------------------------------------------------------------
# Noticing the silence
# ---------------------------------------------------------------------------
@st.fragment(run_every=2.0)
def quiet_watchdog(step: int, deadline: float) -> None:
    """
    Hand over to the quiet prompt when nothing has been recorded in time.

    A fragment, not the `streamlit_autorefresh` the clinician's sketch
    imports, for two reasons. It is first-party, so there is nothing extra to
    install. And it reruns only itself: a whole-script refresh every second
    restarts Annie's prompt video on every tick, so she would never get to
    the end of a sentence.

    Renders nothing. A visible countdown on a clinical screen reads as a test
    the patient is failing.
    """
    if st.session_state.get(f"rec_{step}") is not None:
        return
    if time.monotonic() < deadline:
        return
    set_phase(step, flow.PHASE_SILENCE_RETRY)
    st.rerun(scope="app")


def arm_quiet_watchdog(step: int, settings: Dict[str, object]) -> None:
    """Start, or keep running, the quiet wait for this task."""
    if not settings.get("watch_for_silence", True):
        return
    deadline = st.session_state.silence_deadline.get(step)
    if deadline is None:
        deadline = time.monotonic() + flow.quiet_window_seconds(step)
        st.session_state.silence_deadline[step] = deadline
    quiet_watchdog(step, deadline)


# ---------------------------------------------------------------------------
# Taking a recording
# ---------------------------------------------------------------------------
def handle_recording(step: int, raw: bytes, settings: Dict[str, object]) -> None:
    """
    Judge one submitted recording and route the task to the right phase.

    Three outcomes. Nothing audible arrived, so ask again. The patient asked
    for the instruction instead of attempting the task, so explain it again.
    Or it is usable, so measure it and acknowledge it.

    Every outcome is recorded against a fingerprint of the bytes, because
    `st.audio_input` keeps handing the same file back on every rerun: setting
    a phase and rerunning without that guard sets it again immediately, and
    forever.
    """
    fingerprint = hashlib.sha1(raw).hexdigest()
    if st.session_state.fingerprints.get(step) == fingerprint:
        return
    previous = st.session_state.rejected.get(step)
    if previous is not None and previous[0] == fingerprint:
        return

    audio = None
    decoded = True
    try:
        audio = load_audio(raw)
    except AudioDecodeError:
        decoded = False

    reason = flow.silence_reason(raw, audio, decoded=decoded)
    if reason is not None:
        st.session_state.rejected[step] = (fingerprint, reason)
        log_system(f"task {step} not measured: {reason}")
        set_phase(step, flow.PHASE_SILENCE_RETRY)
        st.rerun()

    # The help scan runs before anything is measured or reported. A
    # three-word "sorry, what do I do?" would otherwise be graded as a
    # thirty-second picture description and pull the fluency scores down.
    transcript = _UNSET
    if step == 4 and settings["stt_on"]:
        transcript = transcribe(audio, model_size=str(settings["model_size"]))
        heard = transcript.text if transcript is not None else None
        if flow.looks_like_a_plea_for_help(heard):
            st.session_state.rejected[step] = (
                fingerprint, "that sounded like a question rather than an answer"
            )
            if heard:
                log_patient(heard)
            log_system(f"task {step}: heard a request for help, not an attempt")
            set_phase(step, flow.PHASE_CLARIFY)
            st.rerun()
        st.session_state.transcript = transcript

    st.session_state.rejected.pop(step, None)
    analyse_task(step, raw, settings, transcript=transcript)
    if st.session_state.results.get(step) is not None:
        set_phase(step, flow.PHASE_AFFIRM)
        st.rerun()


# ---------------------------------------------------------------------------
# What Annie does in each phase
# ---------------------------------------------------------------------------
def render_prompt_phase(
    step: int, script: ui.TaskScript, settings: Dict[str, object]
) -> None:
    """The instruction, as a clip where one exists and as speech where not."""
    if play_clip(script.animation):
        # The clip carries Annie's own voice. The browser speaking over her
        # would make both unintelligible, so the text-to-speech path is only
        # for tasks with no clip.
        return
    if settings["speak"]:
        nonce = f"task{step}"
        if nonce not in st.session_state.spoken:
            components.html(ui.speak_script(script.prompt, nonce), height=0)
            st.session_state.spoken.add(nonce)


def render_affirm_phase(step: int) -> None:
    """Annie acknowledges the recording before the numbers appear."""
    if not play_clip(flow.media_for(step).affirm):
        st.success(
            flow.AFFIRMATION_TEXT.get(step, "Thank you, that is recorded.")
        )


def render_silence_retry_phase(step: int) -> None:
    """The quiet prompt, with a way back to the instruction."""
    play_clip(flow.QUIET_RETRY_AUDIO)
    st.info(
        "It is quiet on my end. Take a breath whenever you are ready, then "
        "record it again."
    )
    reason = rejection_reason(step)
    if reason:
        st.caption(f"What I saw: {reason}.")

    col_a, col_b = st.columns([1, 1], gap="small")
    if col_a.button(
        "I am ready to try again",
        key=f"quiet_ready_{step}",
        type="primary",
        use_container_width=True,
    ):
        set_phase(step, flow.PHASE_PROMPT)
        st.rerun()
    if col_b.button(
        "Explain it again",
        key=f"quiet_explain_{step}",
        type="secondary",
        use_container_width=True,
    ):
        set_phase(step, flow.PHASE_CLARIFY)
        st.rerun()


def render_clarify_phase(
    step: int, script: ui.TaskScript, settings: Dict[str, object]
) -> None:
    """The instruction given again, on request or after a confused answer."""
    media = flow.media_for(step)
    played = play_clip(media.clarify)
    if not played:
        # No clarification clip for this task yet. Replaying the prompt clip
        # is a weaker version of the same thing rather than nothing at all;
        # tasks 1 to 3 get their own clips when the clinician sends them.
        played = play_clip(script.animation)

    reason = rejection_reason(step)
    if reason:
        st.info(f"{reason.capitalize()}. Here is Task {step} again.")
    else:
        st.info(f"Here is Task {step} again.")
    st.markdown(f"**{script.instruction}**")
    st.caption(script.cue_hint)

    if not played and settings["speak"]:
        # The nonce has to change each time or the browser deduplicates the
        # utterance and the repeat is silent.
        count = st.session_state.clarify_count.get(step, 0)
        components.html(
            ui.speak_script(script.prompt, f"clarify{step}-{count}"), height=0
        )

    if st.button(
        "Got it, carry on",
        key=f"clarify_done_{step}",
        type="primary",
    ):
        set_phase(step, flow.PHASE_PROMPT)
        st.rerun()


def mic_context() -> mic.MicContext:
    """What the browser's address says about microphone access."""
    try:
        url = st.context.url
    except Exception:
        # st.context.url arrived in a later Streamlit than the floor we
        # support, and raises outside a script run. No URL means no claim.
        url = None
    return mic.diagnose(url)


def render_mic_warning() -> bool:
    """
    The one microphone problem we can be certain about, stated plainly.

    Returns True when the page cannot use the microphone at all, so callers
    can put the recorder in context rather than leaving a dead button on the
    screen.
    """
    context = mic_context()
    if not context.blocked:
        return False
    st.error(f"**The microphone cannot work on this address.** {context.reason}")
    if context.suggested_url:
        st.markdown(
            f"Open **{context.suggested_url}** instead, on the machine "
            "running the app, and recording will work."
        )
    return True


def render_mic_help() -> None:
    """Everything else that stops a microphone working, in order."""
    with st.expander("Microphone not working?"):
        for n, (title, detail) in enumerate(mic.TROUBLESHOOTING, start=1):
            st.markdown(f"**{n}. {title}**")
            st.markdown(detail)
        st.caption(
            "You can also skip any task. The rest of the check-in still runs "
            "and the task is reported as not attempted."
        )


# ---------------------------------------------------------------------------
# MRI dementia-stage classifier
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_image_model(path: str, mtime: float):
    """
    The classifier, loaded once.

    Keyed on the file's modification time as well as its path, so retraining
    the model and dropping in a new artifact takes effect without restarting
    the app, which is how she works on the triage side.
    """
    return alzheimer.load_model(Path(path))


def render_imaging_view() -> None:
    """
    The MRI tab, restored on 17 Nov at her request and repaired.

    It previously showed "No classifier artifact found. Set
    AURACARE_TRIAGE_REPO to your Cognitive_AI_Triage checkout" and nothing
    else, which is the error she sent back to us. The artifact now ships with
    the project, so there is normally nothing to configure.

    The model is hers, unchanged. This tab only runs it.
    """
    st.markdown("### MRI dementia-stage classifier")
    st.caption(
        "Screening prototype, not a diagnosis. This is the classifier from "
        "the Cognitive_AI_Triage project, running here unchanged."
    )

    artifact = paths.find_image_model()
    if artifact is None:
        st.warning(
            "No classifier artifact found. Expected "
            "`alzheimer_classifier.pkl.gz` in `models/`, or a "
            "Cognitive_AI_Triage checkout beside this project."
        )
        return

    try:
        model = load_image_model(str(artifact), artifact.stat().st_mtime)
    except Exception as exc:
        st.error(f"The classifier artifact could not be loaded: {exc}")
        return

    facts = alzheimer.describe(model)
    upload = st.file_uploader(
        "Upload a brain MRI",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="mri_upload",
    )

    if upload is not None:
        col_img, col_result = st.columns([1, 1], gap="medium")
        with col_img:
            st.image(upload, caption=upload.name, use_container_width=True)
        with col_result:
            prediction = alzheimer.classify(upload, model)
            if not prediction.ok:
                st.error(prediction.error)
            else:
                st.markdown(f"#### {prediction.label}")
                st.progress(min(1.0, prediction.confidence / 100.0))
                st.caption(f"Confidence {prediction.confidence:.1f}%")
                st.markdown("**All classes**")
                for name, value in prediction.probabilities:
                    st.write(f"{name} - {value:.1f}%")

                # The runner-up matters here. On the four labelled images she
                # sent, the Moderate Dementia scan came back Non Demented at
                # 51.5% with the right answer second at 27%. Showing only the
                # top line would hide that the model was close to undecided.
                if len(prediction.probabilities) > 1:
                    gap = (
                        prediction.probabilities[0][1]
                        - prediction.probabilities[1][1]
                    )
                    if gap < 25:
                        st.warning(
                            "The top two classes are close together. Treat "
                            "this one as undecided rather than as a result."
                        )

    with st.expander("About this model"):
        if facts["classes"]:
            st.write("**Classes**: " + ", ".join(facts["classes"]))
        if facts["n_features"]:
            st.write(f"**Features per image**: {facts['n_features']:,}")
        if facts["test_accuracy"] is not None:
            st.write(
                f"**Reported accuracy**: {facts['test_accuracy'] * 100:.1f}% "
                f"test, {facts['train_accuracy'] * 100:.1f}% train"
            )
        st.caption(
            "Those accuracy figures come from the artifact itself and should "
            "be read with care: 100% on training data means the model has "
            "memorised it, and the classes are very unevenly represented "
            "(13,445 Non Demented against 98 Moderate Dementia in the test "
            "set), so a high overall score can hide a weak class. On the "
            "four labelled images supplied on 17 Nov it placed three "
            "correctly and read the Moderate Dementia scan as Non Demented."
        )
        st.caption(f"Artifact: `{artifact}`")


# ---------------------------------------------------------------------------
# Screening flow
# ---------------------------------------------------------------------------
def render_welcome(settings: Dict[str, object]) -> None:
    """
    B3. Annie's greeting, on a screen of its own.

    It used to play inside Task 1, so the first thing the patient saw was two
    Annie videos stacked, the welcome above the task instruction, with the
    recorder pushed below the fold. A greeting is not a task and does not
    belong on the task screen.
    """
    st.markdown(f"### {ui.WELCOME_HEADING}")

    clip = paths.find_animation(paths.INTRO_ANIMATION)
    if clip is not None:
        # The intro is 1184x1344 and portrait, so at full container width it
        # takes over the page. Half the width, centred.
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.video(str(clip), autoplay=True)

    st.markdown(ui.agent_bubble(ui.WELCOME_SCRIPT), unsafe_allow_html=True)

    if (
        clip is None
        and settings["speak"]
        and "welcome" not in st.session_state.spoken
    ):
        components.html(ui.speak_script(ui.WELCOME_SCRIPT, "welcome"), height=0)
        st.session_state.spoken.add("welcome")

    st.caption(
        "Four short tasks using your microphone, then eight questions. You "
        "can skip anything that will not work for you."
    )

    # Said here, before the first task, because it decides whether the whole
    # check-in can work. She found this out by trying to record and getting
    # nothing.
    render_mic_warning()

    if st.button("Begin the check-in", type="primary", key="welcome_begin"):
        st.session_state.welcome_done = True
        st.rerun()


def render_task_step(step: int, settings: Dict[str, object]) -> None:
    script = ui.script_for(step)
    done = {i: i in st.session_state.results for i in range(1, 5)}
    phase = current_phase(step)

    st.markdown(
        ui.progress_dots(step, 4, done)
        + f"&nbsp;&nbsp;<small>Task {step} of 4 &middot; target window "
          f"{script.window}</small>",
        unsafe_allow_html=True,
    )

    # Equal halves rather than 3:2, and a medium gap rather than a large one.
    # At 3:2 with a large gap the clip and the cue panel sat at opposite edges
    # with a wide empty gutter between them, which is what the clinician asked
    # to have closed up on 16 Nov.
    col_chat, col_cue = st.columns([1, 1], gap="medium")

    with col_chat:
        if phase == flow.PHASE_AFFIRM:
            render_affirm_phase(step)
        elif phase == flow.PHASE_SILENCE_RETRY:
            render_silence_retry_phase(step)
        elif phase == flow.PHASE_CLARIFY:
            render_clarify_phase(step, script, settings)
        else:
            render_prompt_phase(step, script, settings)

        # The prompt stays in the log for the clinical view, but only the
        # current one is shown here. Rendering the whole log meant Task 3
        # appeared under Task 1 and Task 2's bubbles and their telemetry
        # lines, which is a transcript, not a conversation - the clinician
        # asked to see one instruction at a time.
        log_agent(script.prompt)
        st.markdown(ui.agent_bubble(script.prompt), unsafe_allow_html=True)

        st.markdown(f"**{script.instruction}**")
        st.caption(
            f"Aim for about {script.record_seconds} seconds. Press record, "
            "complete the task, then press stop."
        )

        render_mic_warning()

        # 2:1 rather than 3:1. With the chat column now half the width
        # instead of three fifths, a quarter of it was too narrow for the
        # button label and "Say that again" wrapped onto two lines.
        col_rec, col_help = st.columns([2, 1], gap="medium")
        with col_rec:
            raw_file = st.audio_input(
                f"Record Task {step}", key=f"rec_{step}",
                label_visibility="collapsed",
            )
        with col_help:
            # Vertical alignment with the recorder, which carries its own
            # label row even when the label is collapsed.
            st.write("")
            if st.button(
                "Say that again",
                key=f"repeat_{step}",
                use_container_width=True,
                type="secondary",
                help="Annie repeats the instruction for this task.",
            ):
                st.session_state.clarify_count[step] = (
                    st.session_state.clarify_count.get(step, 0) + 1
                )
                set_phase(step, flow.PHASE_CLARIFY)
                st.rerun()

        render_mic_help()

        if raw_file is not None:
            handle_recording(step, raw_file.getvalue(), settings)
        elif phase == flow.PHASE_PROMPT and step not in st.session_state.results:
            arm_quiet_watchdog(step, settings)

    with col_cue:
        if script.cue_class:
            st.markdown(ui.cue_html(script), unsafe_allow_html=True)
        else:
            stimulus = paths.find_stimulus()
            if stimulus is not None:
                st.image(str(stimulus), use_container_width=True)
                st.caption(
                    "Describe what you see happening. Original artwork - the "
                    "established clinical cards are under copyright."
                )
            else:
                st.warning(
                    "No stimulus image found. Drop one into `assets/` named "
                    "`stimulus_...` (.png, .jpg, .jpeg or .webp), or generate "
                    "the placeholder with `python scripts/make_stimulus.py`."
                )
        st.caption(f"**Safety check:** {script.safety_check}")

    # -- results for this task ---------------------------------------------
    result = st.session_state.results.get(step)
    if result is None:
        # No usable recording yet. Previously this returned straight away,
        # which also removed the only way forward: with no microphone, or a
        # recording that failed, the patient was left on this task with no
        # button and no explanation. A screening tool must not trap someone
        # on task 1 because one measurement did not work, so the task can be
        # skipped and is then reported as not attempted rather than as a
        # normal result.
        st.caption(
            "Record this task to see your measurements and continue. "
            "If the microphone is not working, you can skip it - the rest of "
            "the check-in still runs, and this task is reported as not "
            "attempted."
        )
        if st.button(
            f"Skip Task {step}",
            key=f"skip_{step}",
            type="secondary",
        ):
            st.session_state.skipped.add(step)
            st.session_state.step = step + 1
            st.rerun()
        return

    quality = st.session_state.quality.get(step)
    if quality is not None:
        for problem in quality.problems:
            st.error(f"Recording problem: {problem}")
        for warning in quality.warnings:
            st.warning(warning)

    st.divider()
    render_task_metrics(step, result)

    for note in getattr(result, "notes", []):
        st.info(note)

    reason = check_intercept(step)
    if reason:
        if step not in st.session_state.intercepts:
            st.session_state.intercepts.append(step)
        st.error(
            f"**Safety intercept — Task {step}.** {reason}\n\n"
            f"{ui.EMERGENCY_PROMPT}"
        )
        if settings["speak"]:
            nonce = f"intercept{step}"
            if nonce not in st.session_state.spoken:
                components.html(ui.speak_script(ui.EMERGENCY_PROMPT, nonce),
                                height=0)
                st.session_state.spoken.add(nonce)
        log_agent(ui.EMERGENCY_PROMPT)

    st.write("")
    col_a, col_b = st.columns([1, 3])
    label = "Continue to the questions" if step == 4 else f"Continue to Task {step + 1}"
    if col_a.button(label, type="primary", use_container_width=True):
        st.session_state.step = step + 1
        st.rerun()
    if reason:
        col_b.caption(
            "You can continue the check-in; the intercept has already been "
            "recorded and appears in the clinical view."
        )


def render_task_metrics(task_id: int, result) -> None:
    """Near-real-time telemetry for the task just recorded."""
    st.markdown(f"##### Task {task_id} measurements")
    assessment = current_assessment()
    task = assessment.task(task_id)

    cols = st.columns(min(4, max(1, len(task.indicators))))
    for i, indicator in enumerate(task.indicators):
        with cols[i % len(cols)]:
            st.metric(
                indicator.label,
                ui.fmt_value(indicator.value, indicator.unit, 2),
            )
            st.markdown(ui.level_pill(indicator.level), unsafe_allow_html=True)
            st.caption(indicator.detail)

    if task_id == 4:
        transcript = st.session_state.transcript
        if transcript is not None and transcript.available:
            with st.expander("Transcript and language flags", expanded=True):
                st.write(transcript.text or "_(no speech transcribed)_")
                flags = analyse_language(transcript.text)
                if flags.matched_terms:
                    st.markdown("**Lexicon matches**")
                    for category, terms in flags.matched_terms.items():
                        st.markdown(f"- `{category}`: {', '.join(terms)}")
                    st.caption(
                        "Matched text is shown so each flag can be checked "
                        "against the transcript. The lexicon cannot tell "
                        "disclosure from quotation or negation."
                    )
                else:
                    st.caption("No lexicon matches.")
        elif transcript is not None:
            st.info(
                f"Transcript unavailable ({transcript.unavailable_reason}). "
                "Acoustic measures above are unaffected."
            )


def render_phq8_intro(settings: Dict[str, object]) -> None:
    """E1. Her introduction to the questions, then a Begin button."""
    st.markdown(f"### {INTRO_HEADING}")

    clip = paths.find_animation(paths.PHQ8_INTRO_ANIMATION)
    if clip is not None:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.video(str(clip), autoplay=True)
            components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)

    st.markdown(ui.agent_bubble(INTRO_SCRIPT), unsafe_allow_html=True)

    # Spoken only where there is no clip, for the same reason as the tasks:
    # Annie's clip carries her own voice and the browser talking over her
    # makes both unintelligible. Her clip arrived on 16 Nov, so this path is
    # now the fallback rather than the normal case.
    if (
        settings["speak"]
        and clip is None
        and "phq_intro" not in st.session_state.spoken
    ):
        components.html(ui.speak_script(INTRO_SCRIPT, "phq_intro"), height=0)
        st.session_state.spoken.add("phq_intro")

    st.caption(
        "Eight questions, one at a time. This is the PHQ-8, a standard "
        "screening questionnaire, scored by simple addition against published "
        "cut-points."
    )
    if st.button("Begin", type="primary", key="phq_begin"):
        st.session_state.phq_intro_done = True
        st.rerun()


def goto_phq8_question(index: int) -> None:
    """Move to a question, and let Annie read it again when it comes round.

    Her script keys the spoken nonce on the index alone, so a question
    revisited with Previous stays silent. Discarding the nonce lets it be read
    again, which is the point of having a voice.
    """
    st.session_state.phq_index = index
    st.session_state.spoken.discard(f"phq_q_{index}")


def save_phq8_answer(key: str, value) -> None:
    """Record one answer. `None` means the patient chose not to answer."""
    responses = dict(st.session_state.phq8_responses)
    responses[key] = value
    st.session_state.phq8_responses = responses


def advance_phq8(index: int) -> None:
    if index < len(PHQ8_ITEMS) - 1:
        goto_phq8_question(index + 1)
    else:
        st.session_state.finished = True
        st.session_state.step = 6
    st.rerun()


def render_phq8_step(settings: Dict[str, object]) -> None:
    """
    E1 to E3: her introduction, one question per screen, read aloud.

    Rebuilt from her `Phq-8v2.txt`, which is the right approach. It drives
    everything from Streamlit state rather than from an HTML carousel. Her
    first version posted answers with `window.parent.postMessage`, which
    `components.html` cannot receive, so no answer would ever have reached
    Python and the screening would have recorded nothing at all.

    Four changes from her script:

    * The heading shows `item.text`, the clinical item wording, not
      `spoken_prompt`. The latter is a spoken sentence, and the first item's
      reads out all four response options, which is useful to hear and wrong
      to look at above four buttons that say the same thing.
    * A question can be skipped. The previous screen let the patient finish
      with items unanswered, and quietly removing that would trap someone who
      does not want to answer one. `score_phq8` already prorates a partial
      response and flags it, or withholds the score below six answers.
    * `phq_index` and `phq_intro_done` are initialised in `init_state`
      instead of lazily, so "Start over" actually starts over.
    * The running total is not shown. That is her decision and it is the
      right one: a live depression score next to the question being answered
      invites the patient to manage the number rather than answer honestly.
    """
    if not st.session_state.phq_intro_done:
        render_phq8_intro(settings)
        return

    # Clamped rather than trusted. A stale `phq_index` from an interrupted
    # session would otherwise index off the end of the list.
    index = max(0, min(int(st.session_state.phq_index), len(PHQ8_ITEMS) - 1))
    item = PHQ8_ITEMS[index]
    answered = st.session_state.phq8_responses.get(item.key)

    st.caption(f"Question {index + 1} of {len(PHQ8_ITEMS)}")
    st.progress((index + 1) / len(PHQ8_ITEMS))
    st.markdown(RECALL_STEM_SINGLE)
    st.markdown(f"### {item.text}")
    st.write("")

    # E3. Annie reads the question in her own voice. Her eight recordings
    # arrived on 16 Nov, so the browser voice is now only the fallback for a
    # question with no clip, and only when prompts are set to be spoken.
    # The player stays on screen so the patient can hear it again.
    if not play_clip(flow.question_audio(item.key)) and settings["speak"]:
        nonce = f"phq_q_{index}"
        if nonce not in st.session_state.spoken:
            components.html(
                ui.speak_script(item.spoken_prompt, nonce), height=0
            )
            st.session_state.spoken.add(nonce)

    # The chosen answer stays highlighted, so a question reached with
    # Previous shows what was said last time rather than looking unanswered.
    for value, label in enumerate(RESPONSE_OPTIONS):
        if st.button(
            label,
            key=f"phq_opt_{index}_{value}",
            use_container_width=True,
            type="primary" if answered == value else "secondary",
        ):
            save_phq8_answer(item.key, value)
            advance_phq8(index)

    st.write("")
    col_back, col_skip = st.columns([1, 1], gap="small")
    if index > 0 and col_back.button(
        "Previous", key="phq_back", use_container_width=True
    ):
        goto_phq8_question(index - 1)
        st.rerun()
    if col_skip.button(
        "Prefer not to answer",
        key=f"phq_skip_{index}",
        use_container_width=True,
    ):
        save_phq8_answer(item.key, None)
        advance_phq8(index)

    st.caption(
        "You can skip a question. The score is then prorated and flagged as "
        "such, or withheld if fewer than six are answered."
    )


def render_completion(settings: Dict[str, object]) -> None:
    """
    The closing screen, worded for the pathway the check-in ended on.

    Built from the clinician's `Ending video_.txt` with four changes, all
    recorded in CLIENT_REQUESTS.md section 11. Her script referenced
    `ui.EMERGENCY_PROPNAT`, which does not exist, so the screen raised
    AttributeError on the emergency pathway alone: it would have failed only
    when a patient was in distress and looked fine in every test that did not
    breach a threshold.
    """
    assessment = current_assessment()
    escalating = assessment.primary_pathway != PATHWAY_ROUTINE

    # F1. Annie's closing clip, chosen by the ending the check-in reached.
    # She sent two on 17 Nov: one that files the results and wishes the
    # patient well, and one that says a clinician should look. Playing the
    # first after an escalation would contradict the banner underneath it,
    # which is the same mistake her original ending script made in writing.
    #
    # Crisis and emergency get no clip on purpose. See `paths` for why.
    urgent_pathways = (
        PATHWAY_EMERGENCY in assessment.pathways
        or PATHWAY_CRISIS in assessment.pathways
    )
    if urgent_pathways:
        closing = None
    elif escalating:
        closing = paths.find_animation(paths.CLOSING_REVIEW_ANIMATION)
    else:
        closing = paths.find_animation(paths.CLOSING_GOOD_ANIMATION)
    if closing is None:
        closing = paths.find_animation(paths.CLOSING_ANIMATION)

    # Held to a third of the width like the welcome: the clips are portrait,
    # and at full container width this one pushes the result of the check-in
    # below the fold.
    if closing is not None:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.video(str(closing), autoplay=True)
            components.html(ui.autoplay_nudge(clip_nonce(closing)), height=0)

    # Her opening line. It goes on routine and clinician review, and not on
    # crisis or emergency. See `ui.SCREENING_THANKS` for why. Keyed off those
    # two pathways rather than off `escalating`, because a clinician review is
    # not an emergency and the warm line reads perfectly well in front of it.
    urgent = (
        PATHWAY_EMERGENCY in assessment.pathways
        or PATHWAY_CRISIS in assessment.pathways
    )
    opener = "" if urgent else ui.SCREENING_THANKS

    if PATHWAY_EMERGENCY in assessment.pathways:
        st.error(f"### ⚠️ {ui.EMERGENCY_PROMPT}")
        spoken = ui.EMERGENCY_PROMPT
    elif PATHWAY_CRISIS in assessment.pathways:
        st.error(f"### {ui.CRISIS_PROMPT}")
        spoken = ui.CRISIS_PROMPT
    elif escalating:
        if opener:
            st.markdown(ui.agent_bubble(opener), unsafe_allow_html=True)
        st.warning(f"### {ui.CLINICIAN_REVIEW_PROMPT}")
        spoken = f"{opener} {ui.CLINICIAN_REVIEW_PROMPT}".strip()
    else:
        # Only say "complete" when nothing is being escalated. The original
        # draft showed this message during an escalation, which contradicted
        # the emergency banner directly above it.
        #
        # And only this branch says the results have been filed. The
        # clinician's draft said "everything is mainly fine" here; a routine
        # pathway means nothing crossed a provisional threshold, not that the
        # patient is well, and the prototype is not entitled to the
        # reassurance. She agreed to this wording on 16 Nov.
        if opener:
            st.markdown(ui.agent_bubble(opener), unsafe_allow_html=True)
        st.success(f"### ✅ {ui.COMPLETION_PROMPT}")
        spoken = f"{opener} {ui.COMPLETION_PROMPT}".strip()

    if settings["speak"] and "completion" not in st.session_state.spoken:
        components.html(ui.speak_script(spoken, "completion"), height=0)
        st.session_state.spoken.add("completion")

    st.caption(
        # The tab strip was shortened to stop the first tab scrolling off the
        # left edge, and this caption was left pointing at the old names.
        "Open the **Results** and **Clinical** tabs above for the full picture."
    )
    if st.button("Start a new check-in"):
        reset_all()
        st.rerun()


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------
def render_patient_view() -> None:
    st.markdown("### Your results")
    if not st.session_state.results and not st.session_state.finished:
        st.info("Complete the check-in first and your results will appear here.")
        return

    assessment = current_assessment()

    if PATHWAY_EMERGENCY in assessment.pathways:
        st.error("**We are routing your assessment to emergency triage.**")
    if PATHWAY_CRISIS in assessment.pathways:
        st.error(f"**Support is available right now.** {CRISIS_RESOURCES_UK}")

    for message in assessment.patient_messages:
        st.write(f"- {message}")

    st.divider()
    phq8 = assessment.phq8
    if phq8 is not None and phq8.scorable:
        st.markdown("#### How you have been feeling")
        st.progress(
            min(1.0, (phq8.total_score or 0) / MAX_SCORE),
            text=f"PHQ-8 {phq8.total_score} of {MAX_SCORE} — {phq8.band}",
        )

    st.markdown("#### What we measured")
    st.caption(
        "Plain-language summary. These are experimental voice indicators, not "
        "a diagnosis."
    )
    friendly = {
        1: "How long you could hold a note, and how steady your voice was",
        2: "How strongly and quickly you could breathe out",
        3: "How fast and evenly you could repeat sounds",
        4: "How you spoke when describing the picture",
    }
    for task in assessment.tasks:
        if not task.measured:
            st.write(f"- **{friendly[task.task_id]}** — could not be measured")
            continue
        st.write(
            f"- **{friendly[task.task_id]}** — "
            f"{'looks typical' if LEVEL_RANK.get(task.level, 0) <= 1 else 'worth a closer look'}"
        )


def render_clinical_view() -> None:
    st.markdown("### Clinical / EHR view")
    if not st.session_state.results and not st.session_state.finished:
        st.info("Complete the check-in first.")
        return

    assessment = current_assessment()
    patient_id = st.session_state.patient_id

    st.markdown(
        f'<div class="sim-banner">{SIMULATION_BANNER}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Overall indicator level", assessment.overall_level)
    col_b.metric("Tasks measured", f"{assessment.n_tasks_measured}/4")
    phq8 = assessment.phq8
    col_c.metric(
        "PHQ-8",
        f"{phq8.total_score}/{MAX_SCORE}" if (phq8 and phq8.scorable) else "not scored",
        phq8.band if (phq8 and phq8.scorable) else None,
    )
    col_d.metric("Pathway", PATHWAY_LABEL[assessment.primary_pathway])

    if assessment.reasons:
        st.markdown("#### Reasons")
        for reason in assessment.reasons:
            st.write(f"- {reason}")

    st.markdown("#### Recommended actions")
    for action in assessment.clinician_actions:
        st.write(f"- {action}")

    st.markdown("#### Indicator table")
    rows = []
    for task in assessment.tasks:
        for indicator in task.indicators:
            rows.append(
                {
                    "Task": task.task_id,
                    "Domain": task.title,
                    "Indicator": indicator.label,
                    "Value": (
                        None if indicator.value is None
                        else round(float(indicator.value), 4)
                    ),
                    "Unit": indicator.unit,
                    "Level": indicator.level,
                    "Threshold": indicator.detail,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        # No task produced a measurable indicator. This happens when every
        # task was skipped, which the check-in allows on purpose so that a
        # broken microphone cannot trap the patient on Task 1. The
        # questionnaire still scores, so the rest of this view is meaningful
        # and only the table has nothing to put in it.
        st.caption(
            "No acoustic indicators were measured. Every task was skipped or "
            "could not be measured, so the assessment below rests on the "
            "questionnaire alone."
        )
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        elevated = frame[frame["Level"].isin(["Elevated", "Critical"])]
        if not elevated.empty:
            st.warning(
                f"{len(elevated)} indicator(s) at Elevated or Critical. "
                "These are provisional prototype thresholds."
            )

    st.markdown("#### Simulated FHIR dispatch")
    bundle = build_bundle(assessment, patient_id)
    # The resource tally that sat here was removed at the clinician's request
    # on 17 Nov. `summarise_bundle` is still exercised by the tests, and the
    # same counts are visible in the bundle itself further down this view.

    comm = next(
        e["resource"] for e in bundle["entry"]
        if e["resource"]["resourceType"] == "CommunicationRequest"
    )
    priority = comm["priority"]
    recipient = comm["recipient"][0]["reference"]
    if priority == "stat":
        st.error(f"**STAT** → `{recipient}`")
    elif priority == "urgent":
        st.warning(f"**URGENT** → `{recipient}`")
    else:
        st.success(f"**ROUTINE** → `{recipient}`")
    st.code(comm["payload"][0]["contentString"], language=None)

    with st.expander("Full FHIR-style bundle"):
        st.json(bundle, expanded=False)
    st.download_button(
        "Download bundle (JSON)",
        data=json.dumps(bundle, indent=2),
        file_name=f"auracare_{patient_id}_bundle.json",
        mime="application/json",
    )

    transcript = st.session_state.transcript
    if transcript is not None and transcript.available:
        with st.expander("Task 4 transcript (verbatim)"):
            st.write(transcript.text)
            st.caption(
                "Always review the transcript directly. The lexicon matches "
                "surface strings and cannot interpret negation, quotation or "
                "hypothetical speech."
            )


    if st.session_state.log:
        with st.expander("Session log"):
            # The check-in screen shows one instruction at a time, so this is
            # the only place the whole exchange can be read: what Annie said,
            # what was recorded, and the telemetry line for each task. It also
            # records the retries. A patient who needed the instruction three
            # times is something a clinician should be able to see.
            for role, body in st.session_state.log:
                who = {"agent": "Annie", "patient": "Patient"}.get(
                    role, "System"
                )
                st.markdown(f"**{who}** · {body}")


def render_reference_view() -> None:
    st.markdown("### Method, thresholds and limitations")

    st.markdown("#### What this prototype does and does not claim")
    st.markdown(
        "- It is a **screening prototype**, not a medical device, and nothing "
        "here is a diagnosis.\n"
        "- Depression severity is scored from the **PHQ-8 by summation "
        "against published cut-points**. No model is involved in that number.\n"
        "- The acoustic measures in Tasks 1-4 are **experimental indicators**. "
        "They are graded against provisional thresholds chosen to drive this "
        "prototype, not against validated clinical cut-offs.\n"
        "- Task 2's measures are acoustic proxies on an uncalibrated "
        "microphone. They are **not spirometry** and no number here is a peak "
        "flow.\n"
        "- All emergency dispatch and FHIR messaging is **simulated**. Nothing "
        "is transmitted to any system or service."
    )

    st.error(
        "**Calibration warning — Task 4 pause ratio.** The protocol "
        "specifies an intercept at 0.45. Measured across 76 thirty-second "
        "windows of the supplied sample recordings, this indicator ranged "
        "0.24–0.72 with a median of **0.48** and exceeded 0.45 in "
        "**57%** of windows. The threshold therefore sits below the median of "
        "ordinary speech and, as specified, cannot separate a struggling "
        "patient from an ordinary one. It is kept as the default because it is "
        "the specified value, and it is tunable in the sidebar. Recalibrate it "
        "against real Task 4 recordings before relying on it — "
        "`scripts/calibrate_pause_ratio.py` reproduces the measurement.\n\n"
        "A breach routes to clinical review, not to the respiratory emergency "
        "pathway: a pause ratio measures language and cognition, so telling "
        "the patient their breathing sounds strained would be wrong."
    )

    st.markdown("#### Provisional thresholds in use")
    rows = []
    for group, items in st.session_state.thresholds.as_dict().items():
        for key, value in items.items():
            rows.append(
                {
                    "Group": group,
                    "Threshold": key,
                    "Current": value,
                    "Default": PROVISIONAL_DEFAULTS[group][key],
                    "Changed": value != PROVISIONAL_DEFAULTS[group][key],
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Published clinical reference values")
    st.caption(
        "Shown for comparison only. Nothing in the app branches on these "
        "numbers; they exist so the distance between a tuned demo threshold "
        "and the literature stays visible."
    )
    for group, entries in CLINICAL_REFERENCE.items():
        with st.expander(group):
            for label, value in entries.items():
                st.write(f"**{label}** — {value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    st.markdown(ui.CSS, unsafe_allow_html=True)

    st.title("\U0001fa7a Auracare: Guided Vocal Biomarkers Screening")
    st.caption(
        "Four-domain bioacoustic check-in with PHQ-8. Screening prototype — "
        "not a diagnosis. Emergency dispatch and FHIR messaging are simulated."
    )

    settings = render_sidebar()

    tab_flow, tab_patient, tab_clinical, tab_mri, tab_ref = st.tabs(
        [
            # Short labels on purpose. Streamlit scrolls the tab strip
            # horizontally when it overflows, and the clinician lost the
            # first tab off the left edge on a narrow window - she could not
            # find the tasks at all. They all have to fit. "MRI" is in that
            # position in the screenshot she sent on 17 Nov asking for the
            # tab back, so it keeps both the name and the place.
            "Check-in",
            "Results",
            "Clinical",
            "MRI",
            "Method",
        ]
    )

    with tab_flow:
        step = int(st.session_state.step)
        if not st.session_state.welcome_done and step <= 4:
            render_welcome(settings)
        elif step <= 4:
            render_task_step(step, settings)
        elif step == 5:
            render_phq8_step(settings)
        else:
            render_completion(settings)

    with tab_patient:
        render_patient_view()
    with tab_clinical:
        render_clinical_view()
    with tab_mri:
        render_imaging_view()
    with tab_ref:
        render_reference_view()


if __name__ == "__main__":
    main()
