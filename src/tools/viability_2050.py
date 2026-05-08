"""Crop viability projection — when does the climate exceed crop tolerance?

Combines CMIP6 climate projections with crop-specific growing seasons
and stage-sensitive temperature thresholds. Critically: thresholds are
applied to seasonal temperature during the relevant growth stage, NOT
annual mean — different crops grow in different seasons.

Currently supports: wheat, rice, maize, cotton, sugarcane.
Sources: FAO ECOCROP, ICAR Package of Practices, PAU agronomy notes.
"""

from typing import TypedDict

from .climate_projection import get_climate_projection


# Crop tolerance with growing season AND key reproductive-stage months.
# Months are 1-12 calendar months covering the *grain-fill / flowering /
# heat-sensitive* window — the bottleneck for that crop.
_CROP_TOLERANCES: dict[str, dict] = {
    "wheat": {
        "season_label": "rabi (winter, sown Oct-Nov, harvested Mar-Apr)",
        "critical_months": [3, 4],  # grain fill — heat-sensitive
        "stage": "grain fill (Mar–Apr)",
        "optimal_temp_c": (15, 22),
        "marginal_temp_c": (22, 27),
        "unviable_temp_c": 30,  # sustained March-April > 30°C means crop failure
        "min_annual_rain_mm": 250,  # below this even with irrigation
        "notes": (
            "Heat above 27°C during grain fill causes 5–10% yield loss per °C. "
            "Above 30°C is non-viable."
        ),
    },
    "rice": {
        "season_label": "kharif (monsoon, Jun-Oct)",
        "critical_months": [7, 8, 9],  # flowering and grain fill
        "stage": "anthesis (flowering)",
        "optimal_temp_c": (22, 28),
        "marginal_temp_c": (28, 35),
        "unviable_temp_c": 38,
        "min_annual_rain_mm": 700,
        "notes": "Heat above 35°C during flowering causes spikelet sterility.",
    },
    "maize": {
        "season_label": "kharif or rabi (multiple seasons)",
        "critical_months": [7, 8],  # silking; assuming kharif planting
        "stage": "silking",
        "optimal_temp_c": (20, 30),
        "marginal_temp_c": (30, 35),
        "unviable_temp_c": 38,
        "min_annual_rain_mm": 350,
        "notes": "Drought during silking is the single largest yield-loss factor.",
    },
    "cotton": {
        "season_label": "kharif (May-Oct)",
        "critical_months": [8, 9],  # boll formation
        "stage": "boll formation",
        "optimal_temp_c": (21, 32),
        "marginal_temp_c": (32, 36),
        "unviable_temp_c": 40,
        "min_annual_rain_mm": 300,
        "notes": "Heat above 35°C during boll formation causes shedding.",
    },
    "sugarcane": {
        "season_label": "year-round (multi-year crop)",
        "critical_months": [4, 5, 6, 7, 8, 9],  # peak growth period
        "stage": "peak growth (Apr-Sep)",
        "optimal_temp_c": (20, 32),
        "marginal_temp_c": (32, 36),
        "unviable_temp_c": 40,
        "min_annual_rain_mm": 800,
        "notes": "Sugarcane needs sustained moisture; rainfall deficit limits viability.",
    },
}


class ViabilityProjection(TypedDict):
    crop: str
    latitude: float
    longitude: float
    season: str
    current_status: str
    projected_2050_status: str
    breach_year: int | None
    critical_temp_2025: float  # avg estimated critical-stage temp 2025–29
    critical_temp_2050: float  # avg estimated critical-stage temp 2046–50
    delta_critical_temp: float
    annual_rain_2025: float
    annual_rain_2050: float
    delta_rain: float
    explanation: str


def _unknown_payload(
    crop_key: str,
    latitude: float,
    longitude: float,
    season: str,
    explanation: str,
) -> ViabilityProjection:
    return {
        "crop": crop_key,
        "latitude": latitude,
        "longitude": longitude,
        "season": season,
        "current_status": "unknown",
        "projected_2050_status": "unknown",
        "breach_year": None,
        "critical_temp_2025": 0.0,
        "critical_temp_2050": 0.0,
        "delta_critical_temp": 0.0,
        "annual_rain_2025": 0.0,
        "annual_rain_2050": 0.0,
        "delta_rain": 0.0,
        "explanation": explanation,
    }


def _classify(seasonal_temp: float, annual_rain: float, tolerance: dict) -> str:
    """Classify viability based on critical-stage temperature and annual rainfall."""
    if (
        seasonal_temp >= tolerance["unviable_temp_c"]
        or annual_rain < tolerance["min_annual_rain_mm"]
    ):
        return "unviable"
    if seasonal_temp > tolerance["marginal_temp_c"][1]:
        return "marginal"
    if tolerance["optimal_temp_c"][0] <= seasonal_temp <= tolerance["optimal_temp_c"][1]:
        return "optimal"
    return "marginal"


def _seasonal_offset_deg_c(tolerance: dict) -> float:
    """Punjab-centric offset from annual mean to critical-stage mean (heuristic)."""
    critical_months: list[int] = tolerance["critical_months"]
    is_warm_season = any(m in (5, 6, 7, 8, 9) for m in critical_months)
    return 6.5 if is_warm_season else 2.0


def _critical_temp_for_row(row: dict, offset: float) -> float:
    return float(row["temperature_2m_mean"]) + offset


_SEVERITY_RANK = {"optimal": 0, "marginal": 1, "unviable": 2}


def _breach_year_vs_baseline(
    yearly_statuses: list[dict[str, int | str]],
    baseline_status: str,
) -> int | None:
    """First year strictly worse than 2025–2029 baseline, if sustained 2+ years."""
    baseline_severity = _SEVERITY_RANK[baseline_status]
    for i, ys in enumerate(yearly_statuses):
        if _SEVERITY_RANK[str(ys["status"])] <= baseline_severity:
            continue
        future = yearly_statuses[i + 1 : i + 3]
        if len(future) < 2:
            continue
        if all(_SEVERITY_RANK[str(f["status"])] > baseline_severity for f in future):
            return int(ys["year"])
    return None


def get_viability_projection(
    crop: str,
    latitude: float,
    longitude: float,
) -> ViabilityProjection:
    """Project viability through 2050 using critical-stage seasonal temperature."""
    crop_key = crop.lower().strip()
    if crop_key not in _CROP_TOLERANCES:
        supported = ", ".join(_CROP_TOLERANCES)
        return _unknown_payload(
            crop_key,
            latitude,
            longitude,
            "",
            f"Crop not supported. Supported crops: {supported}.",
        )

    tolerance = _CROP_TOLERANCES[crop_key]
    season_label = str(tolerance["season_label"])

    try:
        projection = get_climate_projection(latitude, longitude, 2025, 2050)
    except Exception as e:
        return _unknown_payload(
            crop_key,
            latitude,
            longitude,
            season_label,
            f"Could not retrieve climate projection: {e}",
        )

    if not projection:
        return _unknown_payload(
            crop_key,
            latitude,
            longitude,
            season_label,
            "Could not retrieve projection.",
        )

    seasonal_offset = _seasonal_offset_deg_c(tolerance)

    yearly_statuses: list[dict[str, int | str]] = []
    early_temps: list[float] = []
    late_temps: list[float] = []
    early_rain: list[float] = []
    late_rain: list[float] = []

    for row in projection:
        year = int(row["year"])
        critical_temp_estimate = _critical_temp_for_row(row, seasonal_offset)
        annual_rain = float(row["precipitation_sum"])
        yearly_statuses.append(
            {
                "year": year,
                "status": _classify(critical_temp_estimate, annual_rain, tolerance),
            }
        )
        if year <= 2029:
            early_temps.append(critical_temp_estimate)
            early_rain.append(annual_rain)
        if year >= 2046:
            late_temps.append(critical_temp_estimate)
            late_rain.append(annual_rain)

    if not early_temps or not late_temps:
        return _unknown_payload(
            crop_key,
            latitude,
            longitude,
            season_label,
            "Projection series too short for 2025–2029 vs 2046–2050 windows.",
        )

    avg_critical_2025 = sum(early_temps) / len(early_temps)
    avg_critical_2050 = sum(late_temps) / len(late_temps)
    avg_rain_2025 = sum(early_rain) / len(early_rain)
    avg_rain_2050 = sum(late_rain) / len(late_rain)
    d_crit = avg_critical_2050 - avg_critical_2025
    d_rain = avg_rain_2050 - avg_rain_2025

    current_status = _classify(avg_critical_2025, avg_rain_2025, tolerance)
    projected_2050_status = _classify(avg_critical_2050, avg_rain_2050, tolerance)
    breach_year = _breach_year_vs_baseline(yearly_statuses, current_status)

    parts = [
        f"For {crop_key} at ({latitude}, {longitude}) — {season_label}:",
        f"Current (2025-2029): {current_status} — estimated {avg_critical_2025:.1f}°C "
        f"during {tolerance['stage']}, {avg_rain_2025:.0f}mm annual rain.",
        f"Projected (2046-2050): {projected_2050_status} — estimated {avg_critical_2050:.1f}°C "
        f"during {tolerance['stage']}, {avg_rain_2050:.0f}mm annual rain.",
        f"Δ critical temp: {d_crit:+.2f}°C; Δ rain: {d_rain:+.0f}mm.",
    ]
    if breach_year is not None:
        parts.append(
            f"⚠ Viability becomes marginal/unviable starting around {breach_year}."
        )
    elif current_status == "unviable":
        parts.append(
            "Already unviable under the 2025–2029 baseline; breach year only applies "
            "when conditions worsen beyond that baseline."
        )
    else:
        parts.append(f"✓ {crop_key.capitalize()} should remain viable through 2050.")
    parts.append(tolerance["notes"])
    parts.append(
        "Note: critical-stage temperature is estimated from CMIP6 annual mean using a "
        "Punjab seasonal offset. Replace with monthly CMIP6 data for higher precision."
    )
    explanation = " ".join(parts)

    return {
        "crop": crop_key,
        "latitude": latitude,
        "longitude": longitude,
        "season": season_label,
        "current_status": current_status,
        "projected_2050_status": projected_2050_status,
        "breach_year": breach_year,
        "critical_temp_2025": round(avg_critical_2025, 2),
        "critical_temp_2050": round(avg_critical_2050, 2),
        "delta_critical_temp": round(d_crit, 2),
        "annual_rain_2025": round(avg_rain_2025, 0),
        "annual_rain_2050": round(avg_rain_2050, 0),
        "delta_rain": round(d_rain, 0),
        "explanation": explanation,
    }


if __name__ == "__main__":
    print("=== Wheat at Bathinda, Punjab ===")
    proj = get_viability_projection("wheat", 30.21, 74.94)
    print(proj["explanation"])
    print()
    print("=== Rice at Bathinda, Punjab ===")
    proj = get_viability_projection("rice", 30.21, 74.94)
    print(proj["explanation"])
    print()
    print("=== Cotton at Bathinda, Punjab ===")
    proj = get_viability_projection("cotton", 30.21, 74.94)
    print(proj["explanation"])
