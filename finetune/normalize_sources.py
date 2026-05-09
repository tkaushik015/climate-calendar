"""Normalize source field typos and add missing PAU citations.

Run: python3 finetune/normalize_sources.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / "data" / "punjab_agronomy.jsonl"

# Source typo fixes (typo -> canonical)
SOURCE_FIXES = {
    "pau:wheat:rabi_20_26": "pau:wheat:rabi_2025_26",
    "pau:wheat:ri_2025_26": "pau:wheat:rabi_2025_26",
    "u:wheat:rabi_2025_26": "pau:wheat:rabi_2025_26",
    "pau:wheatabi_2025_26": "pau:wheat:rabi_2025_26",
}

# Citation suffixes by category (for entries missing PAU citation)
CITATION_SUFFIX = {
    "wheat": "Source: PAU Package of Practices Rabi 2025-26.",
    "rice": "Source: PAU Package of Practices Kharif 2025.",
    "cotton": "Source: PAU Package of Practices Kharif 2025.",
}


def main() -> None:
    backup = p.with_suffix(".jsonl.bak")
    backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup: {backup}")

    entries = [
        json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    source_fixes = 0
    citation_fixes = 0

    for e in entries:
        # Fix source typos
        if e["source"] in SOURCE_FIXES:
            e["source"] = SOURCE_FIXES[e["source"]]
            source_fixes += 1

        # Add missing PAU citation for non-general entries
        cat = e["category"]
        if cat in CITATION_SUFFIX and "Source: PAU" not in e["output"]:
            suffix = CITATION_SUFFIX[cat]
            if not e["output"].rstrip().endswith("."):
                e["output"] = e["output"].rstrip() + "."
            e["output"] = e["output"] + " " + suffix
            citation_fixes += 1

    p.write_text(
        "\n".join(json.dumps(e, ensure_ascii=True) for e in entries) + "\n",
        encoding="utf-8",
    )
    print(f"Fixed {source_fixes} source typos")
    print(f"Added {citation_fixes} PAU citations")
    print(f"Wrote {len(entries)} entries")


if __name__ == "__main__":
    main()
