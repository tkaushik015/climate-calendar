"""ISRIC SoilGrids — per-GPS soil profile.

Fetches soil properties at a given coordinate from ISRIC's global SoilGrids
database (250 m resolution). Returns key values for the topsoil layer.

If the exact coordinate has no data (some tiles are sparse), automatically
samples the nearest neighbours within a small radius and uses the average.

Source: https://rest.isric.org/soilgrids/v2.0/docs
Free, no API key.
"""

import math
from typing import TypedDict
import requests


class SoilProfile(TypedDict):
    latitude: float
    longitude: float
    depth_cm: str
    sand_percent: float | None
    clay_percent: float | None
    silt_percent: float | None
    soc_g_per_kg: float | None
    ph_water: float | None
    cec_cmol_per_kg: float | None
    bulk_density_kg_per_m3: float | None
    sample_radius_km: float
    explanation: str


_PROPERTIES = {
    "sand": "sand_percent",
    "clay": "clay_percent",
    "silt": "silt_percent",
    "soc": "soc_g_per_kg",
    "phh2o": "ph_water",
    "cec": "cec_cmol_per_kg",
    "bdod": "bulk_density_kg_per_m3",
}

_DEPTH = "0-5cm"
_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Offsets in degrees ≈ 250m at the equator; we sweep up to ~5km
_OFFSETS_DEG = [
    (0.0, 0.0),
    (0.0025, 0.0),
    (-0.0025, 0.0),
    (0.0, 0.0025),
    (0.0, -0.0025),
    (0.005, 0.0),
    (-0.005, 0.0),
    (0.0, 0.005),
    (0.0, -0.005),
    (0.01, 0.0),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.0, -0.01),
    (0.025, 0.0),
    (-0.025, 0.0),
    (0.0, 0.025),
    (0.0, -0.025),
    (0.05, 0.0),
    (-0.05, 0.0),
    (0.0, 0.05),
    (0.0, -0.05),
]


def _query_point(lat: float, lon: float) -> dict[str, float | None]:
    """Query SoilGrids for one point. Returns dict of property→value (None if missing)."""
    params: list[tuple[str, str | float]] = [
        ("lon", lon),
        ("lat", lat),
        ("depth", _DEPTH),
        ("value", "mean"),
    ]
    for prop_key in _PROPERTIES:
        params.append(("property", prop_key))

    r = requests.get(_URL, params=params, timeout=30)
    r.raise_for_status()
    layers = r.json().get("properties", {}).get("layers", [])

    out: dict[str, float | None] = {k: None for k in _PROPERTIES}
    for layer in layers:
        prop_key = layer.get("name")
        if prop_key not in _PROPERTIES:
            continue
        d_factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
        depths = layer.get("depths", [])
        if not depths:
            continue
        mean_val = depths[0].get("values", {}).get("mean")
        if mean_val is None:
            continue
        out[prop_key] = round(mean_val / d_factor, 2)
    return out


def get_soil_profile(latitude: float, longitude: float) -> SoilProfile:
    """Fetch the 0-5 cm topsoil profile for a GPS location.

    If the exact pixel has no data, automatically tries neighbouring pixels
    (up to ~5 km away) and returns the closest valid sample.
    """
    sample_offset = (0.0, 0.0)
    point_data: dict[str, float | None] = {}

    for d_lat, d_lon in _OFFSETS_DEG:
        lat_q = latitude + d_lat
        lon_q = longitude + d_lon
        point_data = _query_point(lat_q, lon_q)
        if any(v is not None for v in point_data.values()):
            sample_offset = (d_lat, d_lon)
            break

    # Approximate offset in km (1° lat ≈ 111 km; lon scaled by cos(lat))
    radius_km = round(
        math.hypot(
            sample_offset[0] * 111,
            sample_offset[1] * 111 * math.cos(math.radians(latitude)),
        ),
        2,
    )

    result: dict = {
        "latitude": latitude,
        "longitude": longitude,
        "depth_cm": _DEPTH.replace("cm", ""),
        "sample_radius_km": radius_km,
    }
    for src_key, out_key in _PROPERTIES.items():
        result[out_key] = point_data.get(src_key)

    sand = result["sand_percent"]
    clay = result["clay_percent"]
    ph = result["ph_water"]
    soc = result["soc_g_per_kg"]

    texture = "unknown"
    if sand is not None and clay is not None:
        if sand > 70:
            texture = "sandy (drains fast, holds little water)"
        elif clay > 35:
            texture = "clay-heavy (holds water, can waterlog)"
        elif clay > 25:
            texture = "loamy clay (good for cereals)"
        else:
            texture = "loamy (well-balanced)"

    ph_note = ""
    if ph is not None:
        if ph < 5.5:
            ph_note = " Soil is acidic — consider lime."
        elif ph > 8.0:
            ph_note = " Soil is alkaline — gypsum or sulfur may help."
        else:
            ph_note = " Soil pH is in the agronomic sweet spot."

    soc_note = ""
    if soc is not None and soc < 10:
        soc_note = " Organic matter is low — prioritize compost / cover crops."

    radius_note = ""
    if radius_km > 0:
        radius_note = f" (Sampled from a point ~{radius_km} km away.)"

    result["explanation"] = (
        f"Topsoil at ({latitude}, {longitude}) is {texture}."
        + ph_note
        + soc_note
        + radius_note
    )

    return result  # type: ignore[return-value]


if __name__ == "__main__":
    soil = get_soil_profile(30.21, 74.94)
    print(soil["explanation"])
    for k, v in soil.items():
        if k != "explanation":
            print(f"  {k}: {v}")
