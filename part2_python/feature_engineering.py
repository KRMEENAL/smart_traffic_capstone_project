"""Add features for Metro_Interstate_Traffic_Volume data. Run: python feature_engineering.py"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/Clean_Metro_Interstate_Traffic_Volume.csv")
FEAT_PATH = Path("data/Features_Metro_Interstate_Traffic_Volume.csv")
logger = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Feature engineering start: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


    for col in ["temp_c"]:
        mn, mx = df[col].min(), df[col].max()
        df[col + "_scaled"] = (df[col] - mn) / (mx - mn) if mx != mn else 0.0
        logger.debug("%s min=%.4f max=%.4f", col, mn, mx)


    logger.info(
        "Feature engineering end: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    return df

# ---weather features engineered --
def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- One-hot encode weather_main (low cardinality: Clear, Clouds, Rain, Snow, etc.) ---
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather", dtype=int)
    df = pd.concat([df, weather_dummies], axis=1)
    logger.info(
        "One-hot encoded weather_main into %s columns: %s",
        weather_dummies.shape[1], list(weather_dummies.columns)
    )

    # --- Derived indicator: is precipitation occurring at all ---
    df["is_precipitation"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)

    # --- Derived indicator: total precipitation (rain + snow combined, mm) ---
    df["total_precipitation"] = df["rain_1h"] + df["snow_1h"]

    # --- Derived indicator: heavily overcast (>75% cloud cover) ---
    df["is_overcast"] = (df["clouds_all"] > 75).astype(int)

    # --- Derived indicator: clear conditions (useful baseline flag) ---
    df["is_clear"] = (df["weather_main"] == "Clear").astype(int)

    # --- Derived indicator: severe weather (thunderstorms, heavy snow/rain descriptions) ---
    severe_keywords = ["thunderstorm", "heavy", "extreme", "squall", "tornado"]
    df["is_severe_weather"] = (
        df["weather_description"]
        .str.lower()
        .str.contains("|".join(severe_keywords), na=False)
        .astype(int)
    )

    logger.info(
        "Weather features engineered: is_precipitation, total_precipitation, "
        "is_overcast, is_clear, is_severe_weather"
    )
    # --- Z-score standardisation (mean = 0, std = 1) ---
    for col in ["temp_c", "traffic_volume"]:
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_zscore"] = (df[col] - mean) / std
        logger.info("%s standardised (z-score): mean=%.2f, std=%.2f", col, mean, std)

    # --- Min-Max scaling (range 0 to 1) ---
    for col in ["temp_c", "traffic_volume"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_minmax"] = (df[col] - col_min) / (col_max - col_min)
        logger.info("%s min-max scaled: min=%.2f, max=%.2f", col, col_min, col_max)

    logger.info(
        "Scaled features engineered: temp_c_zscore, traffic_volume_zscore, "
        "temp_c_minmax, traffic_volume_minmax"
    )
    logger.debug(df[["temp_c", "temp_c_zscore", "temp_c_minmax"]].describe())
    logger.debug(df[["traffic_volume", "traffic_volume_zscore", "traffic_volume_minmax"]].describe())
    return df
#--create congestion category based on traffic volume--
def create_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a data-driven 3-class congestion target from traffic_volume,
    using the 33rd and 66th percentiles of the observed distribution as
    the Low/Medium/High boundaries.
    Rationale:
    - Quantile-based (rather than fixed) thresholds were chosen because
      traffic_volume has no universally agreed "congested" threshold —
      what counts as heavy traffic depends on the road and dataset.
      Using the data's own distribution avoids an arbitrary guess.    
    """
    low_cut = df["traffic_volume"].quantile(1 / 3)
    high_cut = df["traffic_volume"].quantile(2 / 3)

    df["congestion_category"] = pd.cut(
        df["traffic_volume"],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["Low", "Medium", "High"],
    )

    logger.info(
        "Congestion category created (data-driven tertiles): "
        "Low < %.0f <= Medium < %.0f <= High",
        low_cut, high_cut,
    )
    logger.debug("Class distribution:%s", df["congestion_category"].value_counts())
    return df


def main():
    setup_logging()
    try:
        df = pd.read_csv(CLEAN_PATH)
        logger.info("Loaded cleaned data from %s", CLEAN_PATH)
        df = add_features(df)
        df = engineer_weather_features(df)
        df = create_congestion_category(df)
        df.to_csv(FEAT_PATH, index=False)
        logger.info("Saved features to %s", FEAT_PATH)
    except FileNotFoundError:
        logger.error("Cleaned file missing. Run pipeline.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
