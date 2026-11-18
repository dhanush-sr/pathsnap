# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy 1 Hz GPS observations to a road network and reconstructs the driven route.
A candidate classifier scores the nearest road arcs for every GPS point, and an HMM decoded with
Viterbi turns those scores into one connected route. The system runs on two laptops: a FastAPI
inference server (Laptop 1) and a browser UI (Laptop 2).

## Repository layout

| Folder / file | Contents |
|---|---|
| `pathsnap/data.py` | Raw-file loading, cleaning, local metric projection, road-graph construction |
| `pathsnap/features.py` | Route-to-observation label alignment, candidate generation, feature engineering |
| `pathsnap/hmm.py` | HMM emission/transition model, Viterbi decoder, route filling and route metrics |
| `pathsnap/evaluate.py` | Candidate-level metrics (macro F1, precision, recall, log loss, Top-1) |
| `scripts/eda.py` | Dataset inventory, missing-value analysis, trajectory / network plots, Folium EDA map |
| `scripts/build_dataset.py` | Builds candidate feature tables for train and student_test |
| `scripts/train_models.py` | Trains and compares six models (GroupKFold CV + student-test evaluation) |
| `scripts/bayes_opt.py` | Optuna TPE optimisation of LightGBM, tuned vs untuned comparison, final model export |
| `scripts/evaluate_hmm.py` | Decoder tuning on out-of-fold train probabilities, classifier vs HMM comparison |
| `scripts/make_maps.py` | Folium maps (GPS, network, reference, classifier-only and HMM routes) |
| `api/app.py` | FastAPI backend (`/health`, `/trips`, `/predict`) |
| `ui/index.html` | Evaluator interface (static page, Leaflet map) |
| `artifacts/` | `final_model.joblib`, `model_config.json`, `hmm_config.json`, all compared models |
| `results/` | CSV results, experiment logs, plots, Folium maps |
| `report/` | Technical report |

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the dataset next to the code so the partition folders sit at `Student folder/train` and
`Student folder/student_test`. To keep it somewhere else, set `PATHSNAP_DATA` to the folder that
contains the partition folders.

## Reproducing the experiments

Run the scripts in this order from the repository root:

```bash
python scripts/eda.py              # results/eda/*
python scripts/build_dataset.py    # results/features_*.pkl, results/obs_*.csv
python scripts/train_models.py     # results/model_comparison.csv, cv_folds.csv, plots, artifacts/models/
python scripts/bayes_opt.py        # results/bayes_*.{csv,json}, tuned_vs_untuned.csv, artifacts/final_model.joblib
python scripts/evaluate_hmm.py     # artifacts/hmm_config.json, results/hmm_*.csv
python scripts/make_maps.py        # results/maps/trip_*.html
```

All random seeds are fixed to 42.

## Deployment (two laptops)

Both laptops must be on the same network. Find Laptop 1's IP address with `ipconfig` (Windows)
or `ipconfig getifaddr en0` (macOS).

**Laptop 1: API**

Copy the ten evaluation trips into a folder inside the data root, for example
`Student folder/eval_test/<trip_id>/...`. Then start the server:

```bash
export PATHSNAP_PARTITION=eval_test     # default: student_test
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Every request is logged in this terminal with the trip ID, processing time and accuracy.
The `.route` file is read only by the server-side scoring step. The prediction pipeline loads a
trip without its route, and the response never contains the reference route.

**Laptop 2: UI**

```bash
cd ui
python -m http.server 5500
```

1. Open `http://localhost:5500`.
2. Enter `http://<laptop-1-ip>:8000` in the **API endpoint** field and press **Connect**.
3. Pick a trip and press **Run map matching**.

The map tiles need an internet connection.

## API

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | | status, active partition |
| GET | `/trips` | | `[{trip_id, n_points, duration_s}]` |
| POST | `/predict` | `{"trip_id": "00000000"}` | predicted route polyline, per-point matched arc and probability, metrics |

Interactive docs are served at `http://<host>:8000/docs`.
