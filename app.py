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
from auracare.pdf_export import (
    generate_diagnostic_summary,
    get_pdf_download_component,
    mailto_link,
    render_pdf,
    summary_filename,
)

st.set_page_config(
    page_title="Auracare: Multimodal Vocal & Cognitive Health Screening App",
    page_icon="\U0001fa7a",
    layout="wide",
)

ANALYSERS = {
    1: analyse_phonation,
    2: analyse_expiratory,
    3: analyse_ddk,
}


def init_state() -> None:
    defaults = {
        "step": 1,
        "recordings": {},
        "fingerprints": {},
        "results": {},
        "quality": {},
        "transcript": None,
        "phq8_responses": empty_responses(),
        "thresholds": Thresholds(),
        "patient_id": "NHS-948-204-11",
        "log": [],
        "intercepts": [],
        "finished": False,
        "spoken": set(),


        "skipped": set(),


        "phase": flow.PHASE_PROMPT,
        "phase_step": 1,


        "clip_epoch": 0,


        "opening_done": set(),


        "annie_spoke": set(),


        "quiet_spoke": set(),


        "affirm_seen": set(),


        "rejected": {},

        "silence_deadline": {},


        "clarify_count": {},


        "phq_intro_done": False,
        "phq_index": 0,

        "welcome_done": False,


        "summary_attempted": False,
        "diagnostic_summary": None,


        "diagnostic_pdf": None,
        "main_section": "Check-in",
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
        "welcome_done", "opening_done", "affirm_seen", "clip_epoch",
        "annie_spoke", "quiet_spoke",
        "summary_attempted", "diagnostic_summary", "diagnostic_pdf",
    ):
        st.session_state.pop(key, None)
    st.session_state["main_section"] = "Check-in"
    st.session_state["task_nav"] = "Task 1"
    init_state()


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

    move_on_by_itself = st.sidebar.toggle(
        "Move on by itself", value=True, key="auto_advance",
        help="Annie's affirmation moves on to the next task when she "
             "finishes speaking, and her instruction brings up the recorder "
             "when she finishes giving it. Turn this off to step through the "
             "check-in by hand and read each task's measurements.",
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


        rows = "".join(
            '<div class="cut-row"><span>{}</span><span>{}</span></div>'.format(
                html.escape(key.replace("_", " ")), html.escape(str(value))
            )
            for key, value in PROVISIONAL_DEFAULTS["phq8"].items()
        )
        st.markdown(f'<div class="cut-points">{rows}</div>',
                    unsafe_allow_html=True)


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

    st.sidebar.button(
        "\U0001f501 Start over", type="secondary",
        use_container_width=True, on_click=reset_all,
    )

    return {
        "stt_on": stt_on,
        "model_size": model_size,
        "speak": speak,
        "watch_for_silence": watch_for_silence,
        "auto_advance": move_on_by_itself,
    }


_UNSET = object()

def analyse_task(
    task_id: int,
    raw: bytes,
    settings: Dict[str, object],
    transcript=_UNSET,
) -> None:

    fingerprint = hashlib.sha1(raw).hexdigest()
    if st.session_state.fingerprints.get(task_id) == fingerprint:
        return

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

    return assess_screening(
        st.session_state.results.get(1),
        st.session_state.results.get(2),
        st.session_state.results.get(3),
        st.session_state.results.get(4),
        score_phq8(st.session_state.phq8_responses),
        st.session_state.thresholds,
    )

def build_diagnostic_summary() -> None:

    voice = getattr(st.session_state.results.get(1), "voice", None)
    try:
        summary = generate_diagnostic_summary(
            st.session_state.phq8_responses,
            {
                "jitter_local_pct": getattr(voice, "jitter_local_pct", None),
                "shimmer_local_pct": getattr(voice, "shimmer_local_pct", None),
                "hnr_db": getattr(voice, "hnr_db", None),
            },
            patient_id=st.session_state.patient_id,
            assessment=current_assessment(),
        )


        st.session_state.diagnostic_summary = summary
        st.session_state.diagnostic_pdf = render_pdf(summary)
    except Exception as exc:

        log_system(f"summary could not be prepared: {exc}")
        st.session_state.diagnostic_summary = None
        st.session_state.diagnostic_pdf = None

def offer_summary_download(heading: str, key: str) -> None:

    summary = st.session_state.get("diagnostic_summary")
    data = st.session_state.get("diagnostic_pdf")
    if (not summary or not data) and st.session_state.get("finished"):


        if not st.session_state.get("summary_attempted"):
            st.session_state.summary_attempted = True
            build_diagnostic_summary()
            summary = st.session_state.get("diagnostic_summary")
            data = st.session_state.get("diagnostic_pdf")
    if not summary or not data:
        return
    st.markdown(heading)
    st.download_button(
        "Download your summary (PDF)",
        data=data,
        file_name=summary_filename(summary),
        mime="application/pdf",
        key=key,
        use_container_width=True,
    )
    st.caption(
        "One page, yours to keep. It repeats what this prototype is and is "
        "not, because the page will be read away from this screen."
    )
    offer_summary_email(key)

def offer_summary_email(key: str) -> None:

    summary = st.session_state.get("diagnostic_summary")
    if not summary:
        return
    with st.expander("Email these results"):
        address = st.text_input(
            "Send to",
            key=f"{key}_to",
            placeholder="name@example.com",
            help="Left blank, your mail app opens the draft with no "
                 "recipient and you choose one there.",
        )
        st.link_button(
            "Open this in my email",
            mailto_link(summary, address),
            use_container_width=True,
        )
        st.caption(
            "This opens a draft in your own mail app with the results "
            "written in. Nothing is sent by the check-in itself, and nothing "
            "leaves this machine until you press send. An email link cannot "
            "carry a file, so attach the PDF from the button above if you "
            "want the full page."
        )

def check_intercept(task_id: int) -> Optional[str]:

    assessment = current_assessment()
    task = assessment.task(task_id)
    if task is not None and task.intercept_triggered:
        return task.intercept_reason
    return None


def current_phase(step: int) -> str:

    if st.session_state.get("phase_step") != step:
        st.session_state.phase_step = step
        st.session_state.phase = flow.PHASE_PROMPT
        st.session_state.silence_deadline.pop(step, None)
    phase = str(st.session_state.get("phase") or flow.PHASE_PROMPT)
    return phase if phase in flow.PHASES else flow.PHASE_PROMPT

def clip_nonce(clip) -> str:

    return f"{clip.name}#{st.session_state.get('clip_epoch', 0)}"

def bump_clip_epoch() -> None:

    st.session_state.clip_epoch = st.session_state.get("clip_epoch", 0) + 1

def set_phase(step: int, phase: str) -> None:

    st.session_state.phase = phase
    st.session_state.phase_step = step


    bump_clip_epoch()


    st.session_state.silence_deadline.pop(step, None)

def rejection_reason(step: int) -> Optional[str]:

    entry = st.session_state.rejected.get(step)
    return entry[1] if entry else None


AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg", ".opus")

def play_clip(
    filename: Optional[str], *, autoplay: bool = True,
    discreet: bool = False, hand_back: bool = True,
) -> bool:

    clip = paths.find_animation(filename)
    if clip is None:
        return False
    if clip.suffix.lower() in AUDIO_SUFFIXES:
        fmt = "audio/mpeg" if clip.suffix.lower() == ".mp3" else "audio/wav"


        if discreet:


            key = (ui.VOICE_CONTAINER_KEY if hand_back
                   else ui.QUIET_CONTAINER_KEY)
            with st.container(key=key):
                st.audio(str(clip), format=fmt, autoplay=autoplay)
        else:
            st.audio(str(clip), format=fmt, autoplay=autoplay)
        if autoplay:
            components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)
        return True


    st.video(str(clip), autoplay=autoplay)
    if autoplay:
        components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)
    return True


@st.fragment(run_every=2.0)
def quiet_watchdog(step: int, deadline: float) -> None:


    if int(st.session_state.get("step", 0) or 0) != step:
        return
    if st.session_state.get("phase_step") != step:
        return
    if st.session_state.get("phase") != flow.PHASE_PROMPT:
        return
    if st.session_state.get("silence_deadline", {}).get(step) != deadline:
        return
    if step in st.session_state.get("results", {}):
        return

    if step not in st.session_state.get("opening_done", set()):
        return
    if step not in st.session_state.get("annie_spoke", set()):
        return
    if st.session_state.get(f"rec_{step}") is not None:
        return
    if time.monotonic() < deadline:
        return
    set_phase(step, flow.PHASE_SILENCE_RETRY)
    st.rerun(scope="app")

def annie_has_spoken(
    step: int, script: ui.TaskScript, settings: Dict[str, object]
) -> bool:

    if not settings.get("watch_for_silence", True):
        return False
    if step in st.session_state.annie_spoke:
        return True
    clip = paths.find_animation(script.animation)
    if clip is None:
        st.session_state.annie_spoke.add(step)
        return True
    with st.container(key=ui.SPOKE_CONTAINER_KEY):
        if st.button("Annie has started", key=f"spoke_{step}"):
            st.session_state.annie_spoke.add(step)
            st.rerun()
    components.html(
        ui.press_when_clip_plays(clip_nonce(clip), ui.SPOKE_CONTAINER_CLASS),
        height=0,
    )
    return False

def arm_quiet_watchdog(step: int, settings: Dict[str, object]) -> None:

    if not settings.get("watch_for_silence", True):
        return
    deadline = st.session_state.silence_deadline.get(step)
    if deadline is None:
        deadline = time.monotonic() + flow.quiet_window_seconds(step)
        st.session_state.silence_deadline[step] = deadline
    quiet_watchdog(step, deadline)


def handle_recording(step: int, raw: bytes, settings: Dict[str, object]) -> None:

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


def render_prompt_phase(
    step: int, script: ui.TaskScript, settings: Dict[str, object]
) -> None:

    if play_clip(script.animation):


        return
    if settings["speak"]:
        nonce = f"task{step}"
        if nonce not in st.session_state.spoken:
            components.html(ui.speak_script(script.prompt, nonce), height=0)
            st.session_state.spoken.add(nonce)


READY_LABEL = "I am ready"

def next_step_label(step: int) -> str:

    return "Continue to the questions" if step == 4 else f"Continue to Task {step + 1}"

def render_task_controls(step: int, *, forward: str) -> None:

    st.write("")
    with st.container(key="task-footer"):
        widths = [1, 1, 1] if forward != "none" else [1, 1]
        cols = st.columns(widths, gap="small")
        col_back, col_play = cols[0], cols[1]
        col_next = cols[2] if forward != "none" else None
        if col_back.button(
            "‹ Back", key=f"task_back_{step}", use_container_width=True,
        ):
            if step == 1:
                st.session_state.welcome_done = False
            else:
                st.session_state.step = step - 1
                set_phase(step - 1, flow.PHASE_PROMPT)
            st.rerun()
        if col_play.button(
            "▶ Play again", key=f"task_play_{step}",
            use_container_width=True,
        ):
            st.session_state.clarify_count[step] = (
                st.session_state.clarify_count.get(step, 0) + 1
            )
            set_phase(step, flow.PHASE_CLARIFY)
            st.rerun()
        if forward == "next":
            if col_next.button(
                "Next ›", type="primary", use_container_width=True,
                key=f"continue_{step}",
            ):
                leave_task(step)
        elif forward == "skip":
            if col_next.button(
                f"Skip Task {step} ›", key=f"skip_{step}",
                use_container_width=True,
            ):
                st.session_state.skipped.add(step)
                st.session_state.step = step + 1
                st.rerun()

def leave_task(step: int) -> None:

    st.session_state.affirm_seen.add(step)

    st.session_state.silence_deadline.pop(step, None)
    st.session_state.step = step + 1
    set_phase(step + 1, flow.PHASE_PROMPT)
    st.rerun()

def navigate_task_from_menu() -> None:

    selected = str(st.session_state.get("task_nav", "Task 1"))
    target = int(selected.rsplit(" ", 1)[-1])
    if target != int(st.session_state.step):
        st.session_state.step = target
        set_phase(target, flow.PHASE_PROMPT)

def open_results_section() -> None:

    st.session_state.main_section = "Results"

def affirmation_dialog(step: int, settings: Dict[str, object]) -> None:

    played = play_clip(flow.media_for(step).affirm)
    spoken = flow.AFFIRMATION_TEXT.get(step)
    if not played:
        st.success(spoken or "Thank you, that is recorded.")
    elif spoken:


        st.markdown(ui.agent_bubble(spoken), unsafe_allow_html=True)

    label = "Next"
    if st.button(label, type="primary", use_container_width=True,
                 key=f"affirm_next_{step}"):
        leave_task(step)

    if played and settings.get("auto_advance", True):
        clip = paths.find_animation(flow.media_for(step).affirm)
        components.html(
            ui.advance_when_clip_ends(clip_nonce(clip), label), height=0
        )

def quiet_voice_started() -> bool:

    clip = paths.find_animation(flow.QUIET_RETRY_AUDIO)
    if clip is None:

        return True
    nonce = clip_nonce(clip)
    if nonce in st.session_state.quiet_spoke:
        return True
    with st.container(key=ui.QUIET_SPOKE_CONTAINER_KEY):
        if st.button("Quiet prompt started", key=f"quiet_spoke_{nonce}"):
            st.session_state.quiet_spoke.add(nonce)
            st.rerun()
    components.html(
        ui.press_when_quiet_clip_plays(nonce, ui.QUIET_SPOKE_CONTAINER_CLASS),
        height=0,
    )
    return False

def render_silence_retry_phase(step: int) -> None:

    play_clip(flow.QUIET_RETRY_AUDIO, discreet=True, hand_back=False)


    if quiet_voice_started():
        st.info(
            "It is quiet on my end. Take a breath whenever you are ready, "
            "then record it again."
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

    media = flow.media_for(step)


    played = play_clip(media.clarify, discreet=True)
    if not played:


        played = play_clip(script.animation)

    reason = rejection_reason(step)
    if reason:
        st.info(f"{reason.capitalize()}. Here is Task {step} again.")
    else:
        st.info(f"Here is Task {step} again.")
    st.markdown(f"**{script.instruction}**")
    st.caption(script.cue_hint)

    if not played and settings["speak"]:


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

    try:
        url = st.context.url
    except Exception:


        url = None
    return mic.diagnose(url)

def render_mic_warning() -> bool:

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

    with st.expander("Microphone not working?"):
        for n, (title, detail) in enumerate(mic.TROUBLESHOOTING, start=1):
            st.markdown(f"**{n}. {title}**")
            st.markdown(detail)
        st.caption(
            "You can also skip any task. The rest of the check-in still runs "
            "and the task is reported as not attempted."
        )


@st.cache_resource(show_spinner=False)
def load_image_model(path: str, mtime: float):

    return alzheimer.load_model(Path(path))

def render_imaging_view() -> None:

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


def render_welcome(settings: Dict[str, object]) -> None:

    st.markdown(f"### {ui.WELCOME_HEADING}")

    clip = paths.find_animation(paths.INTRO_ANIMATION)
    if clip is not None:


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
        + f"&nbsp;&nbsp;<small>Task {step} of 4</small>",
        unsafe_allow_html=True,
    )


    if phase == flow.PHASE_AFFIRM and step not in st.session_state.affirm_seen:
        affirmation_dialog(step, settings)
        return


    col_chat, col_cue = st.columns([1, 1], gap="medium")


    with col_chat:
        if phase == flow.PHASE_SILENCE_RETRY:
            render_silence_retry_phase(step)
        elif phase == flow.PHASE_CLARIFY:
            render_clarify_phase(step, script, settings)
        else:
            render_prompt_phase(step, script, settings)

        opening = step not in st.session_state.opening_done


        log_agent(script.prompt)
        st.markdown(ui.agent_bubble(script.prompt), unsafe_allow_html=True)

        st.markdown(f"**{script.instruction}**")
        st.caption(
            f"Aim for about {script.record_seconds} seconds. Press record, "
            "complete the task, then press stop."
        )

        render_mic_warning()

        if opening and phase == flow.PHASE_PROMPT:


            if st.button(
                READY_LABEL,
                key=f"ready_{step}",
                type="primary",
                use_container_width=True,
                help="Bring up the recorder without waiting for Annie to "
                     "finish.",
            ):
                st.session_state.opening_done.add(step)
                st.rerun()
            clip = paths.find_animation(script.animation)
            if clip is not None and settings.get("auto_advance", True):
                components.html(
                    ui.advance_when_clip_ends(clip_nonce(clip), READY_LABEL),
                    height=0,
                )
            elif clip is None:

                st.session_state.opening_done.add(step)
                st.rerun()
        else:


            col_rec, col_help = st.columns([2, 1], gap="medium")
            with col_rec:
                raw_file = st.audio_input(
                    f"Record Task {step}", key=f"rec_{step}",
                    label_visibility="collapsed",
                )
            with col_help:


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
            elif (
                phase == flow.PHASE_PROMPT
                and step not in st.session_state.results
                and annie_has_spoken(step, script, settings)
            ):
                arm_quiet_watchdog(step, settings)

    with col_cue:


        supplied = paths.find_task_visual(script.task_id)
        if supplied is not None:


            with st.container(key=ui.CUE_CONTAINER_KEY):
                if supplied.suffix.lower() in (".mp4", ".webm"):


                    st.video(str(supplied), autoplay=True, loop=True,
                             muted=True)
                else:


                    st.image(str(supplied))


            st.markdown(
                f'<div class="cue-title" style="color:{script.colour}">'
                f"{script.cue_label}</div>",
                unsafe_allow_html=True,
            )
            st.caption(script.cue_hint)
        elif script.cue_class:
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


    result = st.session_state.results.get(step)
    if result is None:


        if opening and phase == flow.PHASE_PROMPT:


            render_task_controls(step, forward="none")
            return
        st.caption(
            "Record this task to see your measurements and continue. "
            "If the microphone is not working, you can skip it - the rest of "
            "the check-in still runs, and this task is reported as not "
            "attempted."
        )
        render_task_controls(step, forward="skip")
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

    render_task_controls(step, forward="next")
    if reason:
        st.caption(
            "You can continue the check-in; the intercept has already been "
            "recorded and appears in the clinical view."
        )

def render_task_metrics(task_id: int, result) -> None:

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

    st.markdown(f"### {INTRO_HEADING}")

    clip = paths.find_animation(paths.PHQ8_INTRO_ANIMATION)
    if clip is not None:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.video(str(clip), autoplay=True)
            components.html(ui.autoplay_nudge(clip_nonce(clip)), height=0)

    st.markdown(ui.agent_bubble(INTRO_SCRIPT), unsafe_allow_html=True)


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

    st.session_state.phq_index = index
    st.session_state.spoken.discard(f"phq_q_{index}")

def save_phq8_answer(key: str, value) -> None:

    responses = dict(st.session_state.phq8_responses)
    responses[key] = value
    st.session_state.phq8_responses = responses

def advance_phq8(index: int) -> None:
    if index < len(PHQ8_ITEMS) - 1:
        goto_phq8_question(index + 1)
    else:
        st.session_state.finished = True
        st.session_state.step = 6
        build_diagnostic_summary()
    st.rerun()

def render_phq8_step(settings: Dict[str, object]) -> None:

    if not st.session_state.phq_intro_done:
        render_phq8_intro(settings)
        return


    index = max(0, min(int(st.session_state.phq_index), len(PHQ8_ITEMS) - 1))
    item = PHQ8_ITEMS[index]
    answered = st.session_state.phq8_responses.get(item.key)

    st.caption(f"Question {index + 1} of {len(PHQ8_ITEMS)}")
    st.progress((index + 1) / len(PHQ8_ITEMS))
    st.markdown(RECALL_STEM_SINGLE)
    st.markdown(f"### {item.text}")
    st.write("")


    if not play_clip(flow.question_audio(item.key)) and settings["speak"]:
        nonce = f"phq_q_{index}"
        if nonce not in st.session_state.spoken:
            components.html(
                ui.speak_script(item.spoken_prompt, nonce), height=0
            )
            st.session_state.spoken.add(nonce)


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

    assessment = current_assessment()
    escalating = assessment.primary_pathway != PATHWAY_ROUTINE


    urgent_pathways = (
        PATHWAY_EMERGENCY in assessment.pathways
        or PATHWAY_CRISIS in assessment.pathways
    )
    if PATHWAY_CRISIS in assessment.pathways:
        closing = paths.find_closing_crisis()
    elif PATHWAY_EMERGENCY in assessment.pathways:
        closing = paths.find_closing_emergency()
    else:
        if escalating:
            closing = paths.find_animation(paths.CLOSING_REVIEW_ANIMATION)
        else:
            closing = paths.find_animation(paths.CLOSING_GOOD_ANIMATION)
        if closing is None:
            closing = paths.find_animation(paths.CLOSING_ANIMATION)


    if closing is not None:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.video(str(closing), autoplay=True)
            components.html(ui.autoplay_nudge(clip_nonce(closing)), height=0)


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


        if opener:
            st.markdown(ui.agent_bubble(opener), unsafe_allow_html=True)
        st.success(f"### ✅ {ui.COMPLETION_PROMPT}")
        spoken = f"{opener} {ui.COMPLETION_PROMPT}".strip()

    if settings["speak"] and "completion" not in st.session_state.spoken:
        components.html(ui.speak_script(spoken, "completion"), height=0)
        st.session_state.spoken.add("completion")


    transcript = st.session_state.transcript
    if (
        4 in st.session_state.results
        and transcript is not None
        and not transcript.available
        and PATHWAY_CRISIS not in assessment.pathways
    ):
        st.warning(
            "The spoken part of Task 4 could not be transcribed "
            f"({transcript.unavailable_reason}), so the language check for "
            "self-harm did not run. The acoustic measures are unaffected. "
            "This result should not be read as that check having passed."
        )

    offer_summary_download("#### Your takeaway", "summary_done")

    if st.button(
        "View results", key="completion_view_results", type="primary",
        use_container_width=True,
        on_click=open_results_section,
    ):
        st.rerun()

    st.caption(


        "Open the **Results** and **Clinical** tabs above for the full picture."
    )
    st.button("Start a new check-in", on_click=reset_all)


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

    offer_summary_download("#### Downloadable summary", "summary_results")

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

    offer_summary_download("#### Patient takeaway", "summary_clinical")

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


def main() -> None:
    init_state()
    st.markdown(ui.CSS, unsafe_allow_html=True)


    components.html(ui.keep_videos_inline(), height=0)

    st.title("\U0001fa7a Auracare: Multimodal Vocal & Cognitive Health Screening App")
    st.caption(
        "Four-domain bioacoustic check-in with PHQ-8. Screening prototype — "
        "not a diagnosis. Emergency dispatch and FHIR messaging are simulated."
    )

    settings = render_sidebar()

    tab_names = ["Check-in", "Results", "Clinical", "MRI", "Method"]
    st.session_state.setdefault("main_section", "Check-in")


    if st.session_state.main_section == "Results":
        tab_names = ["Results"] + [n for n in tab_names if n != "Results"]


    panes = dict(zip(tab_names, st.tabs(tab_names)))
    tab_flow = panes["Check-in"]
    tab_patient = panes["Results"]
    tab_clinical = panes["Clinical"]
    tab_mri = panes["MRI"]
    tab_ref = panes["Method"]

    with tab_flow:
        step = int(st.session_state.step)
        if not st.session_state.welcome_done and step <= 4:
            render_welcome(settings)
        elif step <= 4:


            task_labels = [f"Task {i}" for i in range(1, 5)]


            st.session_state["task_nav"] = f"Task {step}"
            st.selectbox(
                "Go to task", task_labels, key="task_nav",
                label_visibility="collapsed",
                on_change=navigate_task_from_menu,
            )
            render_task_step(int(st.session_state.step), settings)
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
