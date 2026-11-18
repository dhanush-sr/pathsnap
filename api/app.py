"""PathSnap inference API (Laptop 1).

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_DATA        dataset root (contains the partition folders)
      PATHSNAP_PARTITION   folder holding the ten evaluation trips (default: student_test)
"""
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, Trip, trip_ids
from pathsnap.features import FEATURES, align_route, build_features
from pathsnap.evaluate import argmax_decode, continuity
from pathsnap.hmm import fill_route, route_scores, viterbi

PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")
RELIABLE_M = 50.0
MODEL = joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib"))
HMM_CFG = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION, "model": type(MODEL).__name__}


@app.get("/trips")
def trips():
    out = []
    for t in trip_ids(PARTITION):
        tr = Trip(PARTITION, t, with_route=False)
        out.append({"trip_id": t, "n_points": len(tr.t), "duration_s": float(tr.t[-1] - tr.t[0])})
    return out


def score(trip_id, hmm_dec, cls_dec, route):
    """Server-side evaluation against the reference route; nothing here is returned raw."""
    ref_trip = Trip(PARTITION, trip_id, with_route=True)
    if ref_trip.route is None:
        return None
    ref, rd = align_route(ref_trip)
    rel = np.where(rd <= RELIABLE_M)[0]
    acc = lambda dec: float(np.mean([dec.get(int(i), -1) == ref[i] for i in rel])) if len(rel) else None
    return {"point_accuracy_hmm": acc(hmm_dec), "point_accuracy_classifier": acc(cls_dec),
            "evaluated_points": int(len(rel)), **route_scores(ref_trip, route, ref_trip.route)}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in trip_ids(PARTITION):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("predict request trip=%s", req.trip_id)
    t0 = time.time()
    trip = Trip(PARTITION, req.trip_id, with_route=False)   # prediction never sees the route file
    df = build_features(trip)
    proba = MODEL.predict_proba(df[FEATURES])[:, 1]
    hmm_dec = viterbi(trip, df, proba, HMM_CFG)
    cls_dec = argmax_decode(df, proba)
    seq = [hmm_dec[k] for k in sorted(hmm_dec)]
    route = fill_route(trip, seq)
    elapsed = time.time() - t0
    metrics = score(req.trip_id, hmm_dec, cls_dec, route)
    metrics = {**(metrics or {}), "route_continuity_hmm": continuity(trip, seq),
               "route_continuity_classifier": continuity(trip, [cls_dec[k] for k in sorted(cls_dec)])}
    pmap = dict(zip(zip(df["obs"].values.tolist(), df["arc"].values.tolist()), proba.tolist()))
    ll = trip.nodes_ll
    path = [[float(ll[trip.arcs[route[0], 0], 1]), float(ll[trip.arcs[route[0], 0], 0])]] if route else []
    path += [[float(ll[trip.arcs[a, 1], 1]), float(ll[trip.arcs[a, 1], 0])] for a in route]
    points = [{"i": i, "t": float(trip.t[i]), "lat": float(trip.track[i, 1]), "lon": float(trip.track[i, 0]),
               "arc": hmm_dec.get(i), "p": round(pmap.get((i, hmm_dec.get(i)), 0.0), 4)}
              for i in range(len(trip.t))]
    log.info("trip=%s points=%d matched_arcs=%d time=%.2fs acc_hmm=%s acc_cls=%s route_f1=%s",
             req.trip_id, len(points), len(route), elapsed,
             _fmt(metrics.get("point_accuracy_hmm")), _fmt(metrics.get("point_accuracy_classifier")),
             _fmt(metrics.get("route_f1")))
    return {"trip_id": req.trip_id, "processing_s": round(elapsed, 3), "n_points": len(points),
            "matched_segments": len(route), "route_path": path, "points": points, "metrics": metrics}


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"
