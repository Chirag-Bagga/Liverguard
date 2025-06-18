# import os
# import re
# import requests
# import joblib
# from typing import Optional
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# # --- CONFIGURATION ---
# LOCAL_URL   = os.getenv("LOCAL_URL", "http://192.168.204.118/")
# MODEL_PATH  = os.getenv("MODEL_PATH", "model.pkl")   # fixed getenv key

# # --- MODEL LOADING ---
# try:
#     model = joblib.load(MODEL_PATH)
# except Exception as e:
#     raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# # --- FASTAPI INIT ---
# app = FastAPI(title="Local Sensor → Model")

# # --- RESPONSE SCHEMAS ---
# class DetailResponse(BaseModel):
#     distance_cm: Optional[float]
#     note:        str

# class PredictionResponse(DetailResponse):
#     prediction: float

# # --- UTIL FUNCTION ---
# def fetch_live_distance() -> tuple[Optional[float], str]:
#     """
#     GET LOCAL_URL, parse out 'Live Distance: <value> cm'
#     Returns (distance_in_cm or None, note_text).
#     """
#     try:
#         resp = requests.get(LOCAL_URL, timeout=3)
#         resp.raise_for_status()
#     except Exception as e:
#         raise HTTPException(status_code=502, detail=f"Error fetching {LOCAL_URL}: {e}")

#     m = re.search(r"Live\s+Distance:\s*(.*?)\s*cm", resp.text, re.IGNORECASE)
#     if not m:
#         raise HTTPException(status_code=500, detail="Could not parse distance from HTML")

#     raw = m.group(1).strip()
#     try:
#         return float(raw), "OK"
#     except ValueError:
#         return None, raw  # e.g. "Out of range"

# # --- ROUTES ---

# @app.get("/details", response_model=DetailResponse)
# def get_details():
#     dist_cm, note = fetch_live_distance()
#     return DetailResponse(distance_cm=dist_cm, note=note)

# @app.get("/predict", response_model=PredictionResponse)
# def predict():
#     dist_cm, note = fetch_live_distance()
#     feature = [dist_cm if dist_cm is not None else -1.0]

#     try:
#         pred = model.predict([feature])[0]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

#     return PredictionResponse(
#         distance_cm = dist_cm,
#         note        = note,
#         prediction  = float(pred)
#     )

import os
import re
import requests
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urljoin

# --- CONFIG ---
LOCAL_URL   = os.getenv("LOCAL_URL", "http://192.168.204.118/")
MODEL_PATH  = os.getenv("MODEL_PATH", "model.pkl")

# --- LOAD MODEL ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")

# --- FASTAPI & SCHEMAS ---
app = FastAPI(title="Local Sensor → Model")

class DetailResponse(BaseModel):
    distance_cm: Optional[float]
    note:        str

class PredictionResponse(DetailResponse):
    prediction: float

# --- FETCH FUNCTION ---
def fetch_live_distance() -> tuple[Optional[float], str]:
    """
    1) Try to GET from common device API endpoints
    2) Fallback to scraping the HTML if no API works
    """
    # 1) Try a few probable JSON/text endpoints that the page's JS might use.
    #    Inspect your browser's DevTools Network tab to see the actual URL;
    #    if yours is different, add it here.
    candidates = ["distance", "api/distance", "sensor", "value"]
    for ep in candidates:
        url = urljoin(LOCAL_URL, ep)
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                txt = r.text.strip()
                # data might be plain number ("10.52") or JSON {"value":10.52}
                # Try plain number first:
                if re.fullmatch(r"\d+(\.\d+)?", txt):
                    return float(txt), "OK"
                # Or try simple JSON:
                m = re.search(r"\"value\"\s*:\s*([\d.]+)", txt)
                if m:
                    return float(m.group(1)), "OK"
        except requests.RequestException:
            pass

    # 2) Fallback: fetch the HTML and regex out a numeric fallback
    try:
        r = requests.get(LOCAL_URL, timeout=3)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach sensor UI: {e}")

    # Look for server-rendered H1 like <h1>Live Distance: 10.52 cm</h1>
    m = re.search(r"Live\s+Distance:\s*([\d.]+)\s*cm", html, re.IGNORECASE)
    if m:
        return float(m.group(1)), "OK"

    # If still no number, capture whatever the JS placeholder shows (e.g. "--" or "Out of range")
    # We try to grab the span content as-is:
    m2 = re.search(r"<span\s+id=['\"]distance['\"]>(.*?)</span>", html, re.IGNORECASE)
    note = m2.group(1).strip() if m2 else "unknown"
    return None, note

# --- ROUTES ---
@app.get("/details", response_model=DetailResponse)
def details():
    dist, note = fetch_live_distance()
    return DetailResponse(distance_cm=dist, note=note)

@app.get("/predict", response_model=PredictionResponse)
def predict():
    dist, note = fetch_live_distance()
    feature = [dist if dist is not None else -1.0]
    try:
        pred = model.predict([feature])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
    return PredictionResponse(distance_cm=dist, note=note, prediction=float(pred))
