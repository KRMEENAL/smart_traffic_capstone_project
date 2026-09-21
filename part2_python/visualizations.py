"""Traffic volume visualisations for Metro Interstate Traffic Volume dataset. Run: python visualizations.py"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FEAT_PATH = Path("data/Features_Metro_Interstate_Traffic_Volume.csv")
FIG_DIR = Path("figures")
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


def main():
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(FEAT_PATH)
        logger.info("Loaded features: %s rows", len(df))

        g = df.groupby("hour")["traffic_volume"].mean()
        p1 = FIG_DIR / "avg_traffic_volume_by_hour.png"
        plt.figure(figsize=(8, 4))
        plt.bar(g.index, g.values, color="teal")
        plt.xlabel("Hour")
        plt.ylabel("Avg traffic volume")
        plt.title("Average traffic volume by hour")
        plt.tight_layout()
        plt.savefig(p1)
        plt.close()
        logger.info("Saved figure: %s", p1)

        g3 = df.groupby("weather_main")["traffic_volume"].mean().sort_values()
        p2 = FIG_DIR / "traffic_weather_relationship.png"
        plt.figure(figsize=(7, 4))
        plt.barh(g3.index, g3.values, color="slateblue")
        plt.xlabel("Avg traffic volume")
        plt.title("Average traffic_volume by weather")
        plt.tight_layout()
        plt.savefig(p2)
        plt.close()
        logger.info("Saved figure: %s", p2)
        
        # --- Temperature vs traffic (scatter — shows the relationship's actual shape) ---
        p3 = FIG_DIR / "temp_vs_traffic.png"
        plt.figure(figsize=(7, 4))
        plt.scatter(df["temp_c"], df["traffic_volume"], alpha=0.1, s=5, color="darkorange")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Traffic volume")
        plt.title("Temperature vs traffic volume")
        plt.tight_layout()
        plt.savefig(p3)
        plt.close()
        logger.info("Saved figure: %s", p3)
        
        # --- Traffic volume distribution (histogram — shows spread/skew directly,
        # which a bar-of-averages chart cannot) ---
        p4 = FIG_DIR / "traffic_volume_distribution.png"
        plt.figure(figsize=(7, 4))
        plt.hist(df["traffic_volume"], bins=50, color="steelblue", edgecolor="white")
        plt.xlabel("Traffic volume")
        plt.ylabel("Frequency")
        plt.title("Distribution of traffic volume")
        plt.tight_layout()
        plt.savefig(p4)
        plt.close()
        logger.info("Saved figure: %s", p4)

        logger.info("All figures created")
    except FileNotFoundError:
        logger.error("Features file missing. Run feature_engineering.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Visualisation step failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
