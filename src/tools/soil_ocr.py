"""Soil-test report OCR — extract structured values from a photographed lab report.

Uses EasyOCR for text extraction (works offline, supports 80+ languages).
A regex-based parser then pulls out the standard agronomic fields:
pH, EC, organic carbon, NPK, zinc.

Source: https://github.com/JaidedAI/EasyOCR
Free, runs locally, no API key.
"""

from typing import Optional, TypedDict
import re

import easyocr


class SoilTestReport(TypedDict, total=False):
    image_path: str
    raw_text: list[str]
    ph: Optional[float]
    ec_ds_per_m: Optional[float]
    organic_carbon_percent: Optional[float]
    available_n_kg_ha: Optional[float]
    available_p_kg_ha: Optional[float]
    available_k_kg_ha: Optional[float]
    zinc_ppm: Optional[float]
    missing_fields: list[str]
    explanation: str


# Lazy-loaded reader
_reader = None


def _get_reader(languages: list[str] = None, gpu: bool = True):
    """Load EasyOCR reader on first use, cache afterwards."""
    global _reader
    if _reader is None:
        if languages is None:
            languages = ["en"]
        _reader = easyocr.Reader(languages, gpu=gpu)
    return _reader


def extract_soil_report(
    image_path: str, languages: list[str] = None, gpu: bool = True
) -> SoilTestReport:
    """Extract structured soil-test values from a photographed report.

    Args:
        image_path: Path to the soil-test report image (JPG, PNG).
        languages: EasyOCR language codes to load. Defaults to ['en'].
        gpu: Whether to use GPU (set False for CPU-only environments).

    Returns:
        Dict with extracted typed fields plus list of fields that OCR
        could not extract (so the caller / Gemma can ask the farmer to
        re-photograph or fill manually).
    """
    reader = _get_reader(languages=languages, gpu=gpu)
    ocr_result = reader.readtext(image_path)

    extracted_lines = [text for _, text, _ in ocr_result]
    blob = " ".join(extracted_lines)

    fields: dict = {}

    m = re.search(r"pH\s*[:;]\s*([\d.]+)", blob, re.IGNORECASE)
    if m:
        fields["ph"] = float(m.group(1))

    m = re.search(r"EC\s*[:;]\s*([\d.]+)", blob, re.IGNORECASE)
    if m:
        fields["ec_ds_per_m"] = float(m.group(1))

    m = re.search(r"Organic\s*Carbon\s*[:;]\s*([\d.]+)", blob, re.IGNORECASE)
    if m:
        fields["organic_carbon_percent"] = float(m.group(1))

    for nutrient, key in [
        ("N", "available_n_kg_ha"),
        ("P", "available_p_kg_ha"),
        ("K", "available_k_kg_ha"),
    ]:
        m = re.search(
            rf"Available\s+{nutrient}\s*[:;]?\s*([\d.]+)", blob, re.IGNORECASE
        )
        if m:
            fields[key] = float(m.group(1))

    m = re.search(r"Zinc\s*[:;]\s*([\d.]+)", blob, re.IGNORECASE)
    if m:
        fields["zinc_ppm"] = float(m.group(1))

    expected = [
        "ph",
        "ec_ds_per_m",
        "organic_carbon_percent",
        "available_n_kg_ha",
        "available_p_kg_ha",
        "available_k_kg_ha",
        "zinc_ppm",
    ]
    missing = [f for f in expected if f not in fields]

    extracted_count = len(expected) - len(missing)
    explanation = (
        f"Extracted {extracted_count} of {len(expected)} expected fields "
        f"from soil-test report at {image_path}."
        + (f" Could not read: {', '.join(missing)}." if missing else "")
    )

    out: SoilTestReport = {
        "image_path": image_path,
        "raw_text": extracted_lines,
        "missing_fields": missing,
        "explanation": explanation,
        **fields,  # type: ignore[misc]
    }
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python soil_ocr.py <image_path>")
        sys.exit(1)
    report = extract_soil_report(sys.argv[1])
    print(report["explanation"])
    for k, v in report.items():
        if k not in ("raw_text", "explanation"):
            print(f"  {k}: {v}")
