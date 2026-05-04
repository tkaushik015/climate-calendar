"""ISRIC SoilGrids — per-GPS soil profile.

Fetches soil properties at a given coordinate from ISRIC's global SoilGrids
database. Returns key values for the topsoil layer.

Hardened for Kaggle / shared-IP environments:
- In-memory cache so repeated queries don't re-hit the API
- Retry-with-backoff on 429 (rate-limit) errors
- Smaller offset sweep (8 nearest pixels) before declaring "no data"
- Pacing delay between calls to stay under SoilGrids' rate limit

Source: https://rest.isric.org/soilgrids/v2.0/docs
Free, no API key.
"""

import math
import time
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

# Smaller offset sweep — covers ~3 km radius in 8 calls, not 21.
# (0,0) first, then 4 cardinal neighbours, then 4 mid-range
_OFFSETS_DEG = [
    (0.0, 0.0),
    (0.005, 0.0),
    (-0.005, 0.0),
    (0.0, 0.005),
    (0.0, -0.005),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 0.02),
    (0.0, -0.02),
]

# In-memory cache for the lifetime of the kernel
_CACHE: dict[tuple[float, float], dict] = {}


def _query_point_with_retry(lat: float, lon: float, max_retries: int = 3) -> dict[str, float | None]:
    """Query SoilGrids for one point, retrying on rate-limit errors."""
    params: list[tuple[str, str | float]] = [
        ("lon", lon),
        ("lat", lat),
        ("depth", _DEPTH),
        ("value", "mean"),
    ]
    for prop_key in _PROPERTIES:
        params.append(("property", prop_key))

    backoff = 1.0
    for attempt in range(max_retries):
        r = requests.get(_URL, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
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
    # All retries exhausted
    return {k: None for k in _PROPERTIES}


def get_soil_profile(latitude: float, longitude: float) -> SoilProfile:
    """Fetch the 0-5 cm topsoil profile for a GPS location.

    If the exact pixel has no data, samples up to 8 neighbouring pixels.
    Caches results so repeated calls for the same coordinate are free.
    """
    cache_key = (round(latitude, 4), round(longitude, 4))
    if cache_key in _CACHE:
        return _CACHE[cache_key]  # type: ignore[return-value]

    sample_offset = (0.0, 0.0)
    point_data: dict[str, float | None] = {k: None for k in _PROPERTIES}

    for d_lat, d_lon in _OFFSETS_DEG:
        lat_q = latitude + d_lat
        lon_q = longitude + d_lon
        point_data = _query_point_with_retry(lat_q, lon_q)
        if any(v is not None for v in point_data.values()):
            sample_offset = (d_lat, d_lon)
            break
        # Pace ourselves between attempts to avoid 429
        time.sleep(0.3)

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

    _CACHE[cache_key] = result
    return result  # type: ignore[return-value]


if __name__ == "__main__":
    soil = get_soil_profile(30.21, 74.94)
    print(soil["explanation"])
    for k, v in soil.items():
        if k != "explanation":
            print(f"  {k}: {v}")
