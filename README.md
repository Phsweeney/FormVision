# FormVision

Computer-vision squat analysis. Upload a back-squat video or turn on your webcam,
and get depth, rep counts, joint angles, and coaching feedback.



![FormVision analysing a set](/example_assets/webcam_tracking.gif)

---

## What it does

Point a camera at a squat and FormVision tells you what happened: how deep you
went, how many reps you completed, how your knees and hips moved through each
one, and which parts of your technique are worth attention.

Every number is defined and inspectable. Depth is a knee angle mapped onto a
configured parallel standard. Where a measurement
could not be taken, it is reported as missing rather than as zero.

It runs in two modes:

- Upload a recorded set and get a full report with a rendered skeleton overlay.
- Turn on your webcam and get live rep counting, depth, and spoken coaching, with
  the entire analysis running in your browser.

---

## Screens

### Upload analysis

![Analysis dashboard](/example_assets/videofeedback.png)


### Live webcam coaching

![Live coaching from a webcam](/example_assets/livewebcam.png)


---

## Features

### Upload mode

- Skeleton overlay rendered back onto your video, with a rep counter and a live
- Per-rep breakdown: depth, timing split into descent and ascent, knee angles per
  side, and whether the hip passed below the knee
- Charts of knee angle, hip angle, and hip height over time, with detected
  repetitions shaded, plus the same traces drawn over the video as it plays
- Coaching feedback with an explanation attached to every item

### Live mode

- Real-time pose tracking at camera frame rate, entirely in the browser
- Rep counting, depth, and tempo as you lift, with a stand-still calibration so
  walking into position does not corrupt the baseline
- Spoken coaching cues through the Web Speech API, with a cooldown so it does not
  talk over itself
- End-of-set summary
- Nothing is uploaded. The video never leaves your machine.

### Model layer (still in progress)

- Trained detectors for knees caving, heels lifting, and left-right asymmetry,
  running alongside the rule-based coaching rather than replacing it
- Model output is always labelled as a model's opinion and never mixed in with
  measurements

---

## Getting started

### Prerequisites

- Python 3.12 or newer
- Node.js 22 or newer
- About 300 MB of disk for dependencies, plus a one-time 6 MB pose model download

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS and Linux
source .venv/bin/activate

pip install -r requirements-dev.txt      # runtime deps plus the test tools
uvicorn app.main:app --reload --port 8000
```

Configuration is optional. Every setting has a working default. To change any of
them, copy `.env.example` to `.env` and edit.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

`npm run dev` automatically stages the MediaPipe runtime and the pose model into
`public/` for live mode. That step is idempotent and non-fatal: if it cannot run,
upload mode still works.

### First run

Upload a back-squat video, or open the live page and allow camera access.

The first upload downloads the MediaPipe pose model (about 6 MB) and caches it in
`backend/models/`, so the first analysis is slower than later ones and needs an
internet connection.

### Filming tips

- Keep your whole body in frame for the entire set
- Film side-on for accurate depth and torso lean
- Film front-on to check left-right balance and knee tracking
- One person in shot, in even lighting

No single camera angle sees everything. See Limitations below.

### Troubleshooting

| Symptom | Cause |
|---|---|
| First analysis fails, later ones work | The pose model download failed. Check the connection and retry. |
| Browser console shows a CORS error | The frontend is on a port other than 3000. Add its origin to `FORMVISION_CORS_ORIGINS` in `backend/.env`. |
| Live mode shows no skeleton | Run `npm run setup:live` and reload. |
| Live mode says "Model unavailable" | Expected if the exported detector bundle is missing. Everything else still works. |
| `No module named pytest` | Install `requirements-dev.txt` rather than `requirements.txt`. |

---

## Configuration

Every threshold in the analysis is a setting, overridable from `backend/.env`.
Nothing is hard-coded in the analysis modules. Examples:

```bash
FORMVISION_PARALLEL_KNEE_ANGLE_DEG=90    # what counts as 100% depth
FORMVISION_GOOD_DEPTH_PERCENT=90         # the threshold for praise
FORMVISION_MAX_TORSO_LEAN_DEG=45         # when to warn about forward lean
FORMVISION_ML_ENABLED=true               # model-derived feedback on or off
```

`GET /config` serves the analysis thresholds to the browser so live mode and the
offline pipeline can never disagree about what "parallel" means.

See `backend/.env.example` for the full annotated list.

---



## Built with

Python, FastAPI, MediaPipe, OpenCV, NumPy, scikit-learn, SQLite, ffmpeg,
TypeScript, Next.js, React, Tailwind, Recharts, Vitest, pytest.

---

## Licence

MIT. See [LICENSE](LICENSE).

The squat pose dataset under `squat_dataset/` is redistributed under CC BY-SA
4.0 and is not covered by the MIT licence. Attribution and terms are in
`squat_dataset/NOTICE`.
