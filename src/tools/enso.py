"""NOAA Oceanic Niño Index (ONI) — current ENSO state.

Fetches the official NOAA ONI history (1950-present) and returns the current
ENSO classification: El Niño / La Niña / Neutral, with intensity.

Source: https://psl.noaa.gov/data/correlation/oni.data
Public domain. No API key.
"""

from typing import TypedDict
import requests


class ENSOState(TypedDict):
    latest_year: int
    latest_month: str
    latest_oni: float
    classification: str   # "El Niño" / "La Niña" / "Neutral"
    intensity: str        # "Weak" / "Moderate" / "Strong" / "Very Strong" / "—"
    explanation: str


_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _classify(oni: float) -> tuple[str, str]:
    """NOAA's ONI classification thresholds."""
    if oni >= 2.0:
        return "El Niño", "Very Strong"
    if oni >= 1.5:
        return "El Niño", "Strong"
    if oni >= 1.0:
        return "El Niño", "Moderate"
    if oni >= 0.5:
        return "El Niño", "Weak"
    if oni <= -2.0:
        return "La Niña", "Very Strong"
    if oni <= -1.5:
        return "La Niña", "Strong"
    if oni <= -1.0:
        return "La Niña", "Moderate"
    if oni <= -0.5:
        return "La Niña", "Weak"
    return "Neutral", "—"


def get_enso_state() -> ENSOState:
    """Fetch the latest ONI value and classify the current ENSO phase.

    Returns:
        Dict with the most recent month's ONI, classification, intensity,
        and a one-sentence farmer-facing explanation.
    """
    url = "https://psl.noaa.gov/data/correlation/oni.data"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    # The PSL ONI file is a fixed-width text file:
    # First line: start_year end_year
    # Then one row per year: YEAR JAN FEB MAR ... DEC
    # Then -99.9 lines for missing future months
    lines = r.text.strip().splitlines()
    start_year, end_year = map(int, lines[0].split())

    latest_year, latest_month_idx, latest_oni = None, None, None
    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        year = int(parts[0])
        if year < start_year or year > end_year:
            continue
        for i, val in enumerate(parts[1:13]):
            v = float(val)
            if v <= -90:   # missing-data sentinel (-99.9)
                continue
            latest_year, latest_month_idx, latest_oni = year, i, v

    if latest_oni is None:
        raise RuntimeError("No valid ONI value found in NOAA file.")

    classification, intensity = _classify(latest_oni)
    month_name = _MONTHS[latest_month_idx]

    explanation = (
        f"As of {month_name} {latest_year}, NOAA's ONI is {latest_oni:+.2f}°C, "
        f"classifying current conditions as {intensity} {classification}."
        if classification != "Neutral"
        else f"As of {month_name} {latest_year}, ENSO is in a Neutral phase "
        f"(ONI {latest_oni:+.2f}°C)."
    )

    return {
        "latest_year": latest_year,
        "latest_month": month_name,
        "latest_oni": round(latest_oni, 2),
        "classification": classification,
        "intensity": intensity,
        "explanation": explanation,
    }


if __name__ == "__main__":
    state = get_enso_state()
    print(state["explanation"])
    print(state)
