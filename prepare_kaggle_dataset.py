"""Prepare trails_sample.csv from the Kaggle trails dataset.

The script expects `kaggle_trails.csv` in the same directory and
produces a balanced, cleaned sample saved as `trails_sample.csv`
with features compatible with the HikeSafe Advisor model.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
RAW_PATH = DATA_DIR / "kaggle_trails.csv"
OUTPUT_PATH = DATA_DIR / "trails_sample.csv"


def map_difficulty(label: str) -> str:
    """Map SAC / text difficulty into green/yellow/red buckets."""
    if not isinstance(label, str):
        return "yellow"

    code = label.lower()

    # Explicit SAC-scale handling first (T1 easiest → T6 toughest).
    if "t1" in code:
        return "green"
    if "t2" in code or "t3" in code:
        return "yellow"
    if "t4" in code or "t5" in code or "t6" in code:
        return "red"

    # Fallback to descriptive keywords if SAC level is absent.
    if any(token in code for token in ("easy", "beginner")):
        return "green"
    if any(token in code for token in ("moderate", "medium", "intermediate")):
        return "yellow"
    if any(token in code for token in ("hard", "difficult", "advanced", "expert")):
        return "red"

    return "yellow"


def flag_exposed_ridge(difficulty_code: str, elevation: float) -> int:
    """Heuristic for exposed ridge indicator."""
    if isinstance(difficulty_code, str):
        code = difficulty_code.lower()
        if any(token in code for token in ("t4", "t5", "t6", "ridge", "arete", "scramble")):
            return 1
    return int(elevation >= 2600)


def flag_slippery_surface(difficulty_code: str, min_temp: float) -> int:
    """Heuristic for slippery / icy conditions."""
    if isinstance(difficulty_code, str):
        code = difficulty_code.lower()
        if any(token in code for token in ("glacier", "ice", "snow", "t5", "t6")):
            return 1
    return int(min_temp < 2)


def clip_percentile(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize a numeric series to mitigate extreme outliers."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def main(sample_size: int = 250) -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw Kaggle dataset not found at {RAW_PATH}. "
            "Download it before running this script."
        )

    df = pd.read_csv(RAW_PATH)

    # Basic integrity filters.
    df = df.drop_duplicates(subset=["name", "length_3d", "moving_time"])
    df = df[
        (df["length_3d"] >= 500)  # >= 0.5 km
        & (df["length_3d"] <= 80000)  # <= 80 km
        & (df["moving_time"] > 0)
        & (df["moving_time"] <= 86400)  # <= 24 h moving time
        & (df["uphill"] >= 0)
    ]
    df = df[df["max_elevation"].notna()]

    df_clean = pd.DataFrame(
        {
            "trail_name": df["name"],
            "distance_km": df["length_3d"] / 1000.0,
            "elevation_gain_m": df["uphill"],
            "max_altitude_m": df["max_elevation"],
            "estimated_duration_h": df["moving_time"] / 3600.0,
            "difficulty_raw": df["difficulty"],
        }
    )

    # Estimate minimum temperature assuming 15°C at sea level with lapse rate 0.0065.
    df_clean["min_temperature_c"] = (
        15.0 - 0.0065 * df_clean["max_altitude_m"]
    ).clip(lower=-15, upper=20)

    df_clean["difficulty_label"] = df_clean["difficulty_raw"].apply(map_difficulty)

    df_clean["exposed_ridge"] = [
        flag_exposed_ridge(diff, elev)
        for diff, elev in zip(df_clean["difficulty_raw"], df_clean["max_altitude_m"])
    ]
    df_clean["slippery_surface"] = [
        flag_slippery_surface(diff, temp)
        for diff, temp in zip(df_clean["difficulty_raw"], df_clean["min_temperature_c"])
    ]

    # Remove rows where we failed to derive a difficulty label.
    df_clean = df_clean[df_clean["difficulty_label"].isin({"green", "yellow", "red"})]

    # Winsorize key numeric features to tame extreme outliers.
    for column in ("distance_km", "elevation_gain_m", "estimated_duration_h"):
        df_clean[column] = clip_percentile(df_clean[column], 0.01, 0.99)

    # Balance classes by random sampling.
    balanced = (
        df_clean.groupby("difficulty_label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), sample_size), random_state=42))
        .reset_index(drop=True)
    )

    # Final column order matches model expectation.
    balanced = balanced[
        [
            "trail_name",
            "distance_km",
            "elevation_gain_m",
            "max_altitude_m",
            "min_temperature_c",
            "exposed_ridge",
            "slippery_surface",
            "estimated_duration_h",
            "difficulty_label",
        ]
    ]

    balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(balanced)} samples to {OUTPUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
