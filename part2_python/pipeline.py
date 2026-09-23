"""Clean Metro Interstate Traffic Volume CSV. Run: python pipeline.py"""

import logging
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",    
]

RAW_PATH = Path("data/Metro_Interstate_Traffic_Volume.csv")
CLEAN_PATH = Path("data/Clean_Metro_Interstate_Traffic_Volume.csv")
logger = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)


def load_raw(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path,na_values=["Nan", "NAN", "nan", "NaN", "None", "none"])
    except FileNotFoundError:
        logger.error("CSV not found: %s", path, exc_info=True)
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error("Could not parse CSV: %s", path, exc_info=True)
        sys.exit(1)
    logger.info("Loaded raw data: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)
    logger.info("Schema validation passed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:    
    before = df["holiday"]
    df["holiday"] = before.str.strip().str.title()
    n_case = int(((df["holiday"] != before) & before.notna()).sum())
    if n_case:
        logger.warning("Standardised holiday casing for %s rows", n_case)
    else:
        logger.info("holiday column already consistent")    

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    bad = int(df["date_time"].isna().sum())
    if bad:
        df = df.dropna(subset=["date_time"]).copy()
        logger.warning("Dropped %s rows with invalid date_time", bad)
    else:
        logger.info("All date_time values parsed OK")
    
    n_dup = int(df.duplicated().sum())    
    #logger.info(df[df.duplicated(keep=False)])
    if n_dup:
        df = df.drop_duplicates().copy()
        logger.warning("Removed %s duplicate rows", n_dup)
    else:
        logger.info("No duplicate rows found") 

    df["month"] = df["date_time"].dt.month
    
    # --- Convert temperature: Kelvin -> Celsius ---
    if "temp" not in df.columns:
        raise KeyError("Expected raw 'temp' (Kelvin) column not found before conversion")
    df["temp_c"] = df["temp"] - 273.15
    df["temp_c"] = df["temp_c"].round(2)
    logger.info("Converted temp (Kelvin) to temp_c (Celsius) for %s rows", len(df))

    # sensor error: temperature < -50
    bad_t = df["temp"] < -50
    n_t = int(bad_t.sum())
    if n_t:
        monthly_median = df.loc[~bad_t].groupby("month")["temp_c"].median()
        fallback = df.loc[~bad_t, "temp_c"].median()
        df.loc[bad_t, "temp_c"] = df.loc[bad_t, "month"].map(monthly_median).fillna(fallback).values
        logger.warning("Imputed %s impossible temp_c values with monthly median", n_t)
    else:
        logger.info("No impossible temp_c values found")
        

    # rain_1h > 200 is not realistic for 1 hour
    bad_r = df["rain_1h"] > 200
    n_r = int(bad_r.sum())
    if n_r:
        for m in df.loc[bad_r, "month"].unique():
            med = df.loc[(df["month"] == m) & (~bad_r), "rain_1h"].median()
            df.loc[bad_r & (df["month"] == m), "rain_1h"] = med
        logger.warning("Imputed %s extreme rain_1h values with monthly median", n_r)
    else:
        logger.info("No extreme rain_1h values found")

    return df
def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # --- Impossible values: traffic_volume cannot be negative ---
    bad_traffic = df["traffic_volume"] < 0
    n_bad_traffic = int(bad_traffic.sum())
    if n_bad_traffic:
        df.loc[bad_traffic, "traffic_volume"] = np.nan
        df["traffic_volume"] = df["traffic_volume"].interpolate()
        logger.warning("Fixed %s negative traffic_volume values via interpolation", n_bad_traffic)
    else:
        logger.info("No negative traffic_volume values found")

    # --- Impossible values: clouds_all must be a 0-100 percentage ---
    bad_clouds = (df["clouds_all"] < 0) | (df["clouds_all"] > 100)
    n_bad_clouds = int(bad_clouds.sum())
    if n_bad_clouds:
        df.loc[bad_clouds, "clouds_all"] = df.loc[bad_clouds, "clouds_all"].clip(0, 100)
        logger.warning("Clipped %s out-of-range clouds_all values to [0, 100]", n_bad_clouds)
    else:
        logger.info("No out-of-range clouds_all values found")

    # --- Impossible values: rain/snow cannot be negative ---
    for col in ["rain_1h", "snow_1h"]:
        bad = df[col] < 0
        n_bad = int(bad.sum())
        if n_bad:
            df.loc[bad, col] = 0
            logger.warning("Set %s negative %s values to 0", n_bad, col)
        else:
            logger.info("No negative %s values found", col)

    # --- Statistical outliers: traffic_volume, via IQR (flag + cap, don't delete) ---
    q1, q3 = df["traffic_volume"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = (df["traffic_volume"] < lower) | (df["traffic_volume"] > upper)
    n_outliers = int(outliers.sum())
    if n_outliers:
        logger.warning(
            "Found %s statistical outliers in traffic_volume (outside [%.0f, %.0f]) — capped, not removed",
            n_outliers, lower, upper,
        )
        df["traffic_volume"] = df["traffic_volume"].clip(lower, upper)
    else:
        logger.info("No statistical outliers found in traffic_volume")

    logger.info("detect_and_handle_outliers finished: %s rows remain", len(df))
    return df
# --- Identify 0 Kelvin temperature readings ---
def temperature_readings(df: pd.DataFrame) -> pd.DataFrame:
    zero_kelvin = df[df["temp"] == 0]
    n_zero_kelvin = len(zero_kelvin)
    logger.info("Found %s rows with temp = 0 Kelvin (sensor error)",n_zero_kelvin)
    #logger.info(zero_kelvin[["date_time", "temp", "traffic_volume"]].head())

    # --- Identify implausible rainfall (> 9,000 mm in a single hour) ---
    extreme_rain = df[df["rain_1h"] > 9000]
    n_extreme_rain = len(extreme_rain)
    logger.info("Found %s rows with rain_1h > 9,000 mm (physically impossible)",n_extreme_rain)
    #logger.info(extreme_rain[["date_time", "rain_1h", "traffic_volume"]].head())
    return df

#---main function invoked first ---
def main():
    setup_logging()
    logger.info("Pipeline started")
    try:
        df = load_raw(RAW_PATH)
        validate_schema(df)
        df = clean_data(df)
        df=detect_and_handle_outliers(df)
        df=temperature_readings(df)
        df.to_csv(CLEAN_PATH, index=False)
        logger.info("Saved cleaned data to %s", CLEAN_PATH)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
