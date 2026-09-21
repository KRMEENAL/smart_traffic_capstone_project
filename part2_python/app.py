"""
CLI for  Metro Interstate Traffic Volume data.

  python app.py summary
  python app.py by-hour --hour 11
  python app.py recommend --day weekend
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

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


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(FEAT_PATH)
        df["date_time"] = pd.to_datetime(df["date_time"])
        return df
    except FileNotFoundError:
        logger.error("Features file not found: %s", FEAT_PATH, exc_info=True)
        sys.exit(1)

# --- Command 1: query traffic for a specific date/time ---
def cmd_query_traffic(args):
    df = load_data()
    mask = df["date_time"].dt.date.astype(str) == args.date
    if args.hour is not None:
        mask &= df["hour"] == args.hour

    result = df.loc[mask]
    if result.empty:
        print(f"No records found for {args.date}" + (f" hour {args.hour}" if args.hour is not None else ""))
        return

    cols = ["date_time", "traffic_volume", "temp_c", "weather_main", "congestion_category"]
    print(result[cols].to_string(index=False))        

#--Command 2 : identify high traffic periods    
def cmd_high_traffic(args):
    df = load_data()
    hourly_avg = df.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False)
    logger.debug(f"DEBUG args:{args}" )  
    logger.debug(f"DEBUG hourly_avg type/len: {len(hourly_avg)}")
    top = hourly_avg.head(args.top)

    logger.info("Top %s highest-traffic hours (by average volume):",args.top)
    for hour, vol in top.items():
        logger.info(f"  Hour {hour:02d}:00 — avg traffic volume: {vol:,.0f}")    

#--Command 3 - retrieve weather and traffic information
def cmd_weather_traffic(args):
    df = load_data()
    if args.weather:
        subset = df[df["weather_main"].str.lower() == args.weather.lower()]
        if subset.empty:
            logger.info(f"No records found for weather condition '{args.weather}'")
            return
        logger.info(f"Weather: {args.weather}")
        logger.info(f"  Records: {len(subset):,}")
        logger.info(f"  Avg traffic volume: {subset['traffic_volume'].mean():,.0f}")
        logger.info(f"  Avg temperature: {subset['temp_c'].mean():.1f}°C")
    else:
        summary = df.groupby("weather_main").agg(
            avg_traffic=("traffic_volume", "mean"),
            avg_temp_c=("temp_c", "mean"),
            records=("traffic_volume", "count"),
        ).sort_values("avg_traffic", ascending=True)
        logger.info(summary.to_string(float_format="%.1f"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini Traffic Analytics Application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("query-traffic", help="Query traffic for a specific date/time")
    p1.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p1.add_argument("--hour", type=int, choices=range(24), help="Hour (0-23), optional")
    p1.set_defaults(func=cmd_query_traffic)

    p2 = subparsers.add_parser("high-traffic", help="Identify high-traffic periods")
    p2.add_argument("--top", type=int, default=5, help="Number of top hours to show (default 5)")
    p2.set_defaults(func=cmd_high_traffic)
    
    p3 = subparsers.add_parser("weather-traffic", help="Retrieve weather and traffic information")
    p3.add_argument("--weather", help="Weather condition (e.g. Clear, Rain, Snow); omit for full breakdown")
    p3.set_defaults(func=cmd_weather_traffic)
    return parser


def main():
    setup_logging()
    parser = build_parser()

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)        
    except FileNotFoundError:
        logger.error("Features file missing. Run feature_engineering.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Command failed: %s", args.command, exc_info=True)
        sys.exit(1) 


if __name__ == "__main__":
    main()
