"""Open-Meteo Historical Weather wrapper.

Pulls per-GPS daily temperature and rainfall from the ERA5 reanalysis (1940-present).
Free, no API key.
"""

from typing import TypedDict
import requests
import pandas as pd


class YearlyClimate(TypedDict):
    year: int
    temperature_2m_mean: float
    precipitation_sum: float


def get_climate_trend(
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
) -> list[YearlyClimate]:
    """Fetch yearly mean temperature and total rainfall for a GPS location.

    Args:
        latitude: WGS84 latitude in degrees.
        longitude: WGS84 longitude in degrees.
        start_year: Inclusive start year (>=1940).
        end_year: Inclusive end year (<=last year).

    Returns:
        List of dicts, one per year, with keys:
            year, temperature_2m_mean (°C), precipitation_sum (mm).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    yearly = (
        df.groupby("year")
        .agg({"temperature_2m_mean": "mean", "precipitation_sum": "sum"})
        .reset_index()
    )
    return yearly.to_dict(orient="records")


if __name__ == "__main__":
    # Smoke test: Bathinda, Punjab — Ramesh Singh's village
    trend = get_climate_trend(30.21, 74.94, 1995, 2024)
    print(f"Got {len(trend)} years of data")
    print(f"First: {trend[0]}")
    print(f"Last: {trend[-1]}")
