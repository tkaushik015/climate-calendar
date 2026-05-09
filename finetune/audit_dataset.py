"""Audit dataset quality. Run: python3 finetune/audit_dataset.py"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = ROOT / "data" / "punjab_agronomy.jsonl"


def main() -> None:
    entries = [
        json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    print(f"Total entries: {len(entries)}\n")

    if not entries:
        print("No entries to audit.")
        return

    # Check 1: Duplicate or near-duplicate instructions
    print("=" * 60)
    print("CHECK 1: Near-duplicate instructions")
    print("=" * 60)
    instr_counts = Counter()
    for e in entries:
        # Normalize for fuzzy matching: lowercase, remove punctuation, collapse spaces
        normalized = re.sub(r"[^\w\s]", "", e["instruction"].lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        instr_counts[normalized] += 1
    duplicates = {k: v for k, v in instr_counts.items() if v > 1}
    print(f"Found {len(duplicates)} near-duplicates")
    for instr, count in list(duplicates.items())[:10]:
        print(f"  {count}x: {instr[:80]}")

    # Check 2: Output length distribution
    print()
    print("=" * 60)
    print("CHECK 2: Output length distribution")
    print("=" * 60)
    lengths = [len(e["output"].split()) for e in entries]
    lengths.sort()
    print(f"Min: {lengths[0]} words")
    print(f"Median: {lengths[len(lengths) // 2]} words")
    print(f"Max: {lengths[-1]} words")
    print(f"Outputs under 20 words: {sum(1 for l in lengths if l < 20)}")
    print(f"Outputs over 200 words: {sum(1 for l in lengths if l > 200)}")

    # Check 3: Outputs not ending in PAU source citation
    print()
    print("=" * 60)
    print("CHECK 3: Missing PAU source citation")
    print("=" * 60)
    missing_source = []
    for i, e in enumerate(entries):
        if "Source: PAU" not in e["output"] and e["category"] != "general":
            missing_source.append((i, e["instruction"][:80], e["category"]))
    print(
        f"Found {len(missing_source)} non-general entries missing 'Source: PAU' citation"
    )
    for i, instr, cat in missing_source[:10]:
        print(f"  [{cat}] {instr}")

    # Check 4: Inconsistent units or numbers
    print()
    print("=" * 60)
    print("CHECK 4: Suspicious number patterns")
    print("=" * 60)
    suspicious = []
    for i, e in enumerate(entries):
        out = e["output"]
        # Look for double-spaces around numbers (paste artifacts)
        if re.search(r"\d  +\d", out):
            suspicious.append((i, "double-space in number", e["instruction"][:60]))
        # Look for mismatched unit pairs like "5 kg/acre 10 quintals"
        if re.search(r"degrees C.*\bdegrees Celsius\b", out) or re.search(
            r"degrees Celsius.*\bdegrees C\b", out
        ):
            suspicious.append(
                (i, "mixed degrees C / degrees Celsius", e["instruction"][:60])
            )
    print(f"Found {len(suspicious)} entries with suspicious patterns")
    for i, issue, instr in suspicious[:10]:
        print(f"  [{issue}] {instr}")

    # Check 5: Source field consistency
    print()
    print("=" * 60)
    print("CHECK 5: Source field values")
    print("=" * 60)
    source_counts = Counter(e["source"] for e in entries)
    for src, n in source_counts.most_common():
        print(f"  {src}: {n}")

    # Check 6: Category-source mismatch
    print()
    print("=" * 60)
    print("CHECK 6: Category-Source mismatch")
    print("=" * 60)
    mismatches = []
    for i, e in enumerate(entries):
        cat = e["category"]
        src = e["source"]
        if cat == "wheat" and "wheat" not in src and "rabi" not in src:
            mismatches.append((i, cat, src, e["instruction"][:60]))
        if cat == "rice" and "rice" not in src:
            mismatches.append((i, cat, src, e["instruction"][:60]))
        if cat == "cotton" and "cotton" not in src:
            mismatches.append((i, cat, src, e["instruction"][:60]))
    print(f"Found {len(mismatches)} category-source mismatches")
    for i, cat, src, instr in mismatches[:10]:
        print(f"  [{cat} -> {src}] {instr}")


if __name__ == "__main__":
    main()
