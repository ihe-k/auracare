# Auracare: Guided Voice Screening Prototype

A working Streamlit prototype that takes a patient through four voice tasks,
records and analyses each one, administers the PHQ-8 conversationally, grades
the results against configurable thresholds, and produces patient-facing and
clinician-facing views plus simulated FHIR-style messaging.

**This is a screening prototype, not a medical device and not a diagnosis. All
emergency dispatch and FHIR messaging is simulated — nothing is transmitted to
any system or service.**

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Point at the Cognitive_AI_Triage checkout for the model artifacts and
# sample recordings (optional — the voice flow works without them).
export AURACARE_TRIAGE_REPO=/path/to/Cognitive_AI_Triage

python scripts/make_stimulus.py     # generates the Task 4 picture
streamlit run app.py
```

Open <http://localhost:8501>. Allow microphone access when the browser asks.

---

## What it does

### The four tasks

| Task | Domain | Record | What is measured |
|---|---|---|---|
| 1 | Respiratory & laryngeal | 10 s | Sustained phonation time, jitter, shimmer, HNR, pitch variability |
| 2 | Expiratory airway | 10 s | Exhalation duration, rise time, decay slope, tonal-peak (wheeze) prominence |
| 3 | Neuromotor speech | 10 s | Syllable repetition rate, inter-syllable timing variability |
| 4 | Cognitive & language | 30 s | Pause ratio, speech and articulation rate, lexical diversity, negative-language flags |

Each task shows the spoken agent prompt from the protocol, an animated visual
pacing cue, and (optionally) reads the prompt aloud using the browser's own
speech synthesis — no API key, no network call.

Analysis runs as soon as each recording stops and the measurements appear
immediately beneath the task. With transcription enabled, Task 4 takes roughly
as long as the recording to process.

### PHQ-8

The eight standard items are asked conversationally, scored by summation, and
banded against the published cut-points (Kroenke et al., 2009). Unanswered
items are prorated when at least six are answered and flagged as prorated;
below six the total is withheld rather than guessed. Missing items are never
treated as zero, because that biases a screening score toward under-referral.

### Risk and routing

Escalation is a *set* of pathways rather than one value, because a patient can
simultaneously trip the respiratory intercept and disclose self-harm, and those
route to different places:

- **Respiratory emergency** — Task 1 or Task 2 breach. Stat dispatch.
- **Mental health crisis** — self-harm language in the transcript. Stat dispatch
  and crisis resources shown to the patient. This fires independently of the
  PHQ-8 score, because PHQ-8 omits the self-harm item entirely.
- **Urgent clinical review** — PHQ-8 in the severe band, or a Task 4 pause-ratio
  breach.
- **Routine clinical review** — PHQ-8 ≥ 10, or any elevated acoustic indicator.

### Views

- **Check-in** — the guided flow.
- **Your results** — plain-language patient view with support messages and,
  for high scores, professional-help and crisis guidance.
- **Clinical / EHR view** — indicator table, reasons, recommended actions, and
  the simulated FHIR bundle (QuestionnaireResponse, Observations, RiskAssessment,
  CommunicationRequest), downloadable as JSON.
- **MRI classifier** — the corrected image pipeline (see below).
- **Method & limits** — thresholds in use, published reference values, and the
  calibration warnings.

---

## Testing

```bash
python -m pytest tests/ -q          # 228 tests
```

The acoustic tests check measurements against signals whose true values are
known by construction, not against previous runs:

- **Phonation duration** recovers exactly across 1.0–8.0 s (error < 0.001 s).
- **DDK rate** recovers across 2–8 syll/s to within 0.1%. This catches the
  `N/duration` vs `(N-1)/span` bias, which is ~5% at a 4-second task — enough
  to push a healthy speaker across a 5 Hz threshold.
- **Pause ratio** matches construction to within 0.02, and correctly excludes
  leading and trailing silence.
- **Jitter** reads < 0.05% on a perfectly periodic signal and scales linearly
  with injected perturbation.
- **Wheeze index** separates broadband breath (5.1 dB) from a 700 Hz tone
  (19.4 dB) and localises the peak to 703 Hz.
- **App rendering** runs the real `app.py` through Streamlit's `AppTest`
  harness, including both reported UI defects.

### Verification scripts

```bash
python scripts/verify_image_fix.py        # buggy vs corrected image pipeline
python scripts/verify_severity_model.py   # PHQ-8 sensitivity sweep
python scripts/verify_audio_pipeline.py   # end to end, routine and emergency
python scripts/calibrate_pause_ratio.py   # threshold calibration on real speech
```

---

## Layout

```
app.py                      Streamlit UI: guided flow and the result views
auracare/
  config.py                 provisional thresholds + clinical reference values
  audio_io.py               decode, resample, quality gate
  vad.py                    energy voice-activity detection
  stt.py                    pluggable transcription + negative-language lexicon
  phq8.py                   the instrument, scoring, banding
  risk.py                   indicator grading, pathway routing, messages
  fhir.py                   simulated FHIR-style resources
  severity_model.py         optional adapter for the repo model (experimental)
  paths.py                  locating artifacts and the sibling repo
  ui.py                     protocol script, CSS, animations, speech synthesis
  features/
    voice.py                Parselmouth: jitter, shimmer, HNR, pitch
    phonation.py            Task 1
    expiratory.py           Task 2
    ddk.py                  Task 3 (+ shared syllable detector)
    fluency.py              Task 4
  imaging/alzheimer.py      corrected MRI pipeline (+ the buggy one, for proof)
tests/                      228 tests
scripts/                    verification and asset generation
assets/stimulus_park.png    Task 4 picture stimulus
```

---

## Configuration

| Variable | Purpose |
|---|---|
| `AURACARE_TRIAGE_REPO` | Path to the Cognitive_AI_Triage checkout (models, samples) |
| `AURACARE_MODEL_DIR` | Alternative directory holding the `.pkl` artifacts |

Thresholds are editable in the sidebar, exportable to JSON and re-importable.
The provisional thresholds are kept strictly separate from the published
clinical reference values, which are display-only and never branched on — so a
tuned demo number can never be mistaken for a validated cut-off.

Models download on first use and cache (`tiny.en` ≈ 75 MB, `base.en` ≈ 145 MB).
Transcription takes roughly as long as the recording on CPU.

---

## Known limitations

- The negative-language lexicon matches literal strings. It cannot interpret
  negation ("I am *not* hopeless" flags), quotation, or attribution ("my brother
  said he wants to die" flags), and it misses any paraphrase not enumerated. The
  transcript is always shown in full so every flag can be checked. Inflections
  in the safety-critical category are enumerated explicitly, with a test.
- Task 2's measures are acoustic proxies on an uncalibrated consumer
  microphone. They are not spirometry and nothing here is a peak flow.
- Acoustic thresholds beyond the pause ratio are engineering values, not
  validated against a clinical population.
- `st.audio_input` records a complete take and analyses it on stop. This is
  near-real-time per task, not a continuous sub-300 ms streaming pipeline.
- PHQ-8 omits the self-harm item by design, so the score alone cannot detect
  suicidal ideation. The transcript lexicon partially covers that gap and routes
  independently, but a direct risk assessment is still required.
