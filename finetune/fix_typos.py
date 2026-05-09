"""Fix corrupted JSON keys in punjab_agronomy.jsonl from earlier paste-corruption.

The keys got mangled (categy -> category, soue -> source, etc.)
Run: python3 finetune/fix_typos.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / "data" / "punjab_agronomy.jsonl"

# Map of (corrupted_key -> correct_key)
KEY_FIXES = {
    "t": "input",
    "categy": "category",
    "cegory": "category",
    "soue": "source",
    "sour": "source",
    "soce": "source",
    "oput": "output",
}

# Lines to delete entirely (line 478 has an extra "invariant" field plus the regular fields,
# so it's a different kind of corruption - we'll handle that separately)
DELETE_LINES_BY_INSTRUCTION_PREFIX = [
    "What is sheath blight in rice?",
]

# The replacement entry for the deleted "What is sheath blight in rice?" line
# (this is its corrected version with proper schema)
REPLACEMENT_ENTRIES = [
    {
        "instruction": "What is sheath blight in rice?",
        "input": "",
        "output": "Sheath blight is a fungal disease of rice caused by Rhizoctonia solani. Greyish green lesions with purple margin develop on the leaf-sheath above the water level. As lesions enlarge and coalesce, they girdle the stem and disrupt nutrient flow. Severe attack causes poor filling of grains and heavy yield loss. The disease appears typically at the maximum tillering to boot stage. Source: PAU Package of Practices Kharif 2025.",
        "category": "rice",
        "source": "pau:rice:kharif_2025",
    },
]


def main() -> None:
    # Backup
    backup = p.with_suffix(".jsonl.bak")
    backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup saved: {backup}")

    # Read all entries
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    fixed_entries = []
    deleted_count = 0
    fixed_key_count = 0

    for line in lines:
        obj = json.loads(line)
        instruction = obj.get("instruction", "")

        # Should we delete this entry entirely?
        if any(instruction.startswith(prefix) for prefix in DELETE_LINES_BY_INSTRUCTION_PREFIX):
            deleted_count += 1
            continue

        # Apply key renames
        new_obj = {}
        for k, v in obj.items():
            if k in KEY_FIXES:
                new_obj[KEY_FIXES[k]] = v
                fixed_key_count += 1
            else:
                new_obj[k] = v

        fixed_entries.append(new_obj)

    # Add replacement entries
    for entry in REPLACEMENT_ENTRIES:
        if not any(e["instruction"] == entry["instruction"] for e in fixed_entries):
            fixed_entries.append(entry)

    # Validate that every entry now has the correct schema
    expected = {"instruction", "input", "output", "category", "source"}
    problems = []
    for i, entry in enumerate(fixed_entries):
        missing = expected - set(entry.keys())
        extra = set(entry.keys()) - expected
        if missing or extra:
            problems.append((i, missing, extra, entry.get("instruction", "")[:80]))

    if problems:
        print(f"\nWARNING: {len(problems)} entries still have schema issues:")
        for i, missing, extra, instr in problems:
            print(f"  Entry {i}: missing={missing}, extra={extra}")
            print(f"    {instr}")
        print("\nNot writing changes. Investigate above.")
    else:
        p.write_text(
            "\n".join(json.dumps(e, ensure_ascii=True) for e in fixed_entries) + "\n",
            encoding="utf-8",
        )
        print(f"\nFixed {fixed_key_count} corrupted keys")
        print(f"Deleted {deleted_count} bad entries (replaced with corrected versions)")
        print(f"Wrote {len(fixed_entries)} clean entries to {p}")


if __name__ == "__main__":
    main()
