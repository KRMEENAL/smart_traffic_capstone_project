"""Task 6.3-6.5: FastAPI deployment mock-up + monitoring/alerting.

Serves the "production" traffic-volume regressor chosen in the Task 6 notebook
(sections 6.1-6.2), and exposes a simple drift-based monitoring/alerting check.

Run standalone:
    uvicorn deployment_api:app --reload
    (then POST to http://127.0.0.1:8000/predict, see /docs for the schema)

Or, as the Task 6 notebook does it, import `app` directly and drive it with
fastapi.testclient.TestClient - no server process or port needed, fully
offline and deterministic, which is why the notebook demos it that way.

Expects three files in the current working directory, all produced by running
the Task 6 notebook (sections 6.1-6.2) first:
    production_model.joblib   - the trained sklearn model (joblib.dump)
    model_metadata.json       - feature_cols, weather category mapping, version info
    reference_stats.json      - monitoring baseline (reference feature samples + MAE)
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.metrics import mean_absolute_error

MODEL_PATH = Path("production_model.joblib")
METADATA_PATH = Path("model_metadata.json")
REFERENCE_STATS_PATH = Path("reference_stats.json")

for path in (MODEL_PATH, METADATA_PATH, REFERENCE_STATS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found in the current working directory. Run the Task 6 notebook "
            "(sections 6.1-6.2) first, from this same directory - it trains the production "
            "model and writes this file."
        )

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())
reference_stats = json.loads(REFERENCE_STATS_PATH.read_text())

FEATURE_COLS = metadata["feature_cols"]
WEATHER_CATEGORY_TO_COLUMN = metadata["weather_category_to_column"]

app = FastAPI(
    title="Metro Traffic Volume Predictor",
    description="Deployment mock-up serving the Task 1/6 traffic-volume regressor.",
    version=str(metadata["model_version"]),
)


class PredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday ... 6=Sunday")
    is_holiday: bool = False
    weather_main: str = Field(..., description=f"One of: {sorted(WEATHER_CATEGORY_TO_COLUMN)}")
    temp_c: float
    clouds_all: float = Field(..., ge=0, le=100)
    rain_1h: float = Field(0.0, ge=0)
    snow_1h: float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float
    model_version: str
    model_description: str
    predicted_at: str


class MonitoringRecord(PredictionRequest):
    # Optional: include the true traffic_volume once it's known, to also check
    # prediction-error drift. Omit it (on every record) to check feature drift only.
    actual_traffic_volume: Optional[float] = None


class MonitoringRequest(BaseModel):
    records: List[MonitoringRecord]


def build_feature_row(payload: "PredictionRequest") -> dict:
    """Raw, human-friendly API input -> the engineered feature row the model expects.

    Mirrors the exact feature engineering used in Task 1 / Task 4 / Task 5
    (cyclical hour/day-of-week encoding, one-hot weather category, derived
    weather indicators) so the model sees the same kind of input it was
    trained on, without the API contract leaking that internal representation.
    """
    weather_col = WEATHER_CATEGORY_TO_COLUMN.get(payload.weather_main)
    if weather_col is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown weather_main {payload.weather_main!r}. "
                f"Known categories: {sorted(WEATHER_CATEGORY_TO_COLUMN)}"
            ),
        )

    row = {c: 0 for c in FEATURE_COLS}
    row["hour"] = payload.hour
    row["day_of_week"] = payload.day_of_week
    row["is_weekend"] = int(payload.day_of_week >= 5)
    row["is_holiday"] = int(payload.is_holiday)
    row["hour_sin"] = math.sin(2 * math.pi * payload.hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * payload.hour / 24)
    row["day_of_week_sin"] = math.sin(2 * math.pi * payload.day_of_week / 7)
    row["day_of_week_cos"] = math.cos(2 * math.pi * payload.day_of_week / 7)
    row[weather_col] = 1
    row["is_precipitation"] = int(payload.rain_1h > 0 or payload.snow_1h > 0)
    row["total_precipitation"] = payload.rain_1h + payload.snow_1h
    row["is_overcast"] = int(payload.clouds_all > 75)
    row["is_clear"] = int(payload.weather_main == "Clear")
    # weather_description isn't part of this API's contract, so is_severe_weather is
    # approximated from weather_main alone (categories that are inherently severe),
    # rather than the keyword match on weather_description used at training time.
    # This is a deliberate simplification of the training-time feature - documented
    # here rather than silently diverging from it.
    row["is_severe_weather"] = int(payload.weather_main in {"Thunderstorm", "Squall"})
    row["clouds_all"] = payload.clouds_all
    row["temp_c"] = payload.temp_c
    return row


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"]}


@app.get("/model/info")
def model_info():
    return {
        "model_version": metadata["model_version"],
        "model_description": metadata["model_description"],
        "holdout_MAE": metadata["holdout_MAE"],
        "holdout_R2": metadata["holdout_R2"],
        "n_features": len(FEATURE_COLS),
        "weather_categories": sorted(WEATHER_CATEGORY_TO_COLUMN),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    row = build_feature_row(payload)
    X = pd.DataFrame([row])[FEATURE_COLS]
    prediction = float(model.predict(X)[0])
    return PredictionResponse(
        predicted_traffic_volume=round(prediction, 1),
        model_version=str(metadata["model_version"]),
        model_description=metadata["model_description"],
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )


def _check_drift(current_df: pd.DataFrame, has_actuals: bool) -> dict:
    """Shared by /monitoring/status and the notebook's own drift demos (6.4/6.5).

    Feature distribution drift: a two-sample Kolmogorov-Smirnov test per monitored
    feature, comparing the incoming batch to a stored reference sample - flags a
    feature whose distribution has measurably shifted (p < 0.05).

    Prediction error drift: only checked when every record includes the (now-known)
    actual traffic_volume - compares this batch's MAE to the model's reference
    (holdout) MAE, and flags a meaningful degradation (> 1.3x).
    """
    checks = {}
    for feat, ref_sample in reference_stats["feature_samples"].items():
        if feat not in current_df.columns:
            continue
        cur_vals = current_df[feat].dropna().values
        if len(cur_vals) < 5:
            continue
        stat_, pvalue = stats.ks_2samp(ref_sample, cur_vals)
        alert = bool(pvalue < reference_stats["thresholds"]["ks_pvalue"])
        checks[f"feature_drift_{feat}"] = {
            "status": "ALERT" if alert else "PASS",
            "ks_statistic": round(float(stat_), 4),
            "p_value": round(float(pvalue), 5),
            "detail": (
                f"{feat} distribution {'differs from' if alert else 'consistent with'} "
                f"reference (KS p={pvalue:.4f})"
            ),
        }

    if has_actuals:
        preds = model.predict(current_df[FEATURE_COLS])
        mae = mean_absolute_error(current_df["actual_traffic_volume"], preds)
        ratio = mae / reference_stats["reference_mae"]
        alert = bool(ratio > reference_stats["thresholds"]["mae_ratio"])
        checks["prediction_error_drift"] = {
            "status": "ALERT" if alert else "PASS",
            "current_mae": round(float(mae), 2),
            "reference_mae": round(reference_stats["reference_mae"], 2),
            "ratio": round(float(ratio), 3),
            "detail": (
                f"current MAE {mae:.1f} vs reference {reference_stats['reference_mae']:.1f} "
                f"(x{ratio:.2f})"
            ),
        }

    overall = "ALERT" if any(c["status"] == "ALERT" for c in checks.values()) else "PASS"
    return {"overall_status": overall, "checks": checks}


@app.post("/monitoring/status")
def monitoring_status(payload: MonitoringRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must be a non-empty list")

    rows = []
    has_actuals = all(r.actual_traffic_volume is not None for r in payload.records)
    for record in payload.records:
        row = build_feature_row(record)
        if record.actual_traffic_volume is not None:
            row["actual_traffic_volume"] = record.actual_traffic_volume
        rows.append(row)

    current_df = pd.DataFrame(rows)
    report = _check_drift(current_df, has_actuals=has_actuals)
    report["n_records"] = len(payload.records)
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    return report
