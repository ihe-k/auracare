# Auracare: Vocal Biomarker & Digital Triage Prototype

Auracare is a Streamlit-based digital triage prototype that guides a user through four structured vocal tasks and an optional PHQ-8 self-report questionnaire to extract acoustic, articulatory, and fluency biomarkers alongside simulated FHIR-style clinical summaries.

[Link to App](https://auracare-bjocdkbkznmm2j7yc8h5sf.streamlit.app/)

> **Disclaimer:** This software is a research and engineering prototype for educational and demonstration purposes only. It is **not** a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## What it does

1. **Guided Vocal Protocol (4 Tasks):**
   * **Task 1 — Sustained Phonation:** Measures vocal fold stability, fundamental frequency ($F_0$), jitter, shimmer, and harmonics-to-noise ratio (HNR).
   * **Task 2 — Expiratory Control:** Evaluates respiratory support and maximum phonation/fricative duration.
   * **Task 3 — Diadochokinetic Rate (DDK):** Assesses articulatory motor speed, syllable rate, and regularity.
   * **Task 4 — Connected Speech (Picture Description):** Analyses speech fluency, speaking rate, pause patterns, hesitation markers, and optional lexical sentiment.
2. **Multimodal Guidance:** Each step provides pre-rendered visual prompt animations (`intro.mp4`, `task1.mp4`–`task4.mp4`), on-screen pacing cues, and optional browser speech synthesis.
3. **PHQ-8 Screening & Risk Stratification:** Combines acoustic indicators with optional PHQ-8 responses to grade risk indicators and route users to appropriate informational pathways.
4. **FHIR-Style Output:** Generates simulated HL7 FHIR-compatible JSON bundles (`Patient`, `Observation`, `QuestionnaireResponse`, and `RiskAssessment` resources) for interoperability testing.
5. **Reference MRI Pipeline (`auracare/imaging/`):** Includes a standalone diagnostic comparison module demonstrating data-leakage and preprocessing corrections for structural MRI classification.

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

---

## Testing

The repository includes a comprehensive automated test suite covering feature extraction, hesitation analysis, animation assets, risk routing, FHIR serialization, and regression checks.

Run the full suite with:

```bash
pytest
```

*(Note: Run `pytest --collect-only -q` to verify the exact active test count in your environment.)*

---

## Layout

```text
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
    hesitation.py           pause and hesitation analysis
  imaging/alzheimer.py      corrected MRI pipeline (+ the buggy one, for proof)
tests/                      automated test suite (unit, integration, and regression tests)
scripts/                    verification and asset generation
assets/
  animations/               guided intro and task prompt videos (intro.mp4, task1-4.mp4)
  stimulus_park.png         Task 4 picture stimulus
```

---

## Known limitations

* **Turn-based recording:** `st.audio_input` records a complete take and analyses it on stop. This is near-real-time per task, not a continuous sub-300 ms streaming pipeline.
* **Provisional thresholds:** Acoustic cutoff values in `auracare/config.py` are heuristic baselines calibrated for prototype demonstration rather than validated clinical norms.
* **Microphone variance:** Consumer laptop and mobile microphones introduce spectral and amplitude differences that can affect absolute shimmer and HNR readings.

---

## Future roadmap

* **Real-time WebRTC streaming:** Migrate from turn-based browser recording (`st.audio_input`) to a low-latency WebRTC audio pipeline for continuous voice-activity detection, live patient pacing feedback, and telehealth integration.
* **Multi-disease vocal biomarker library:** Build a consented, de-identified repository of standardized multi-task vocal samples across respiratory, neurological, and psychiatric conditions to train and validate generalized cross-condition screening models.
* **Clinical model retraining:** Implement subject-level train/test splits and retrain multi-stage classifiers on standardised clinical cohorts.
