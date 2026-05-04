"""Open-Meteo Climate API — CMIP6 climate projections to 2050.

Fetches downscaled future climate projections for any GPS location.
Returns yearly mean temperature and total rainfall.

Default model is MRI-AGCM3-2-S (high-resolution Japanese climate model).
Source: https://open-meteo.com/en/docs/climate-api
Free, no API key.
"""

from typing import TypedDict
import requests
import pandas as pd


class YearlyProjection(TypedDict):
    year: int
    temperature_2m_mean: float
    precipitation_sum: float


def get_climate_projection(
    latitude: float,
    longitude: float,
    start_year: int = 2025,
    end_year: int = 2050,
    model: str = "MRI_AGCM3_2_S",
) -> list[YearlyProjection]:
    """Fetch downscaled CMIP6 climate projection for a GPS location."""
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "models": model,
        "daily": "temperature_2m_mean,precipitation_sum",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    daily = r.json()["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year

    temp_col = next((c for c in df.columns if c.startswith("temperature_2m_mean")), None)
    rain_col = next((c for c in df.columns if c.startswith("precipitation_sum")), None)
    if not temp_col or not rain_col:
        raise RuntimeError(f"Unexpected schema: {df.columns.tolist()}")

    yearly = (
        df.groupby("year")
        .agg({temp_col: "mean", rain_col: "sum"})
        .rename(columns={temp_col: "temperature_2m_mean", rain_col: "precipitation_sum"})
        .reset_index()
    )
    yearly = yearly.round(2)
    return yearly.to_dict(orient="records")


if __name__ == "__main__":
    proj = get_climate_projection(30.21, 74.94, 2025, 2050)
    print(f"Got {len(proj)} years of projections")
    print(f"2025: {proj[0]}")
    print(f"2050: {proj[-1]}")
    diff = proj[-1]["temperature_2m_mean"] - proj[0]["temperature_2m_mean"]
    print(f"Projected ∆T (2025 → 2050): {diff:+.2f}°C")
