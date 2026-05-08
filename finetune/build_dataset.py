"""Build the Punjab agronomy instruction-tuning dataset.

Aggregates Q&A-style examples from PAU Package of Practices, ICAR-IIWBR
advisories, and other sources into a single JSONL ready for Unsloth
LoRA fine-tuning of Gemma 4 E4B.
"""

from pathlib import Path
import json


# Each example: (instruction, input, output, source_tag)
WHEAT_EXAMPLES: list[tuple[str, str, str, str]] = [
    (
        "I am a wheat farmer in Bathinda, Punjab. My soil test shows pH 8.2 and "
        "zinc 0.62 ppm (low). What zinc application is recommended?",
        "",
        "For alkaline Punjab soils (pH > 8) with low zinc, apply 25 kg of zinc "
        "sulphate (ZnSO4) per acre at sowing, broadcast and incorporated into "
        "the topsoil. Pre-soak wheat seeds in 0.5% ZnSO4 solution for 12 hours "
        "before sowing for early uptake. Without zinc, expect 15-25% yield "
        "reduction on alkaline Punjab soils. Source: PAU Package of Practices.",
        "pau:wheat:nutrients",
    ),
    (
        "When should I plant wheat in Bathinda this year?",
        "",
        "The recommended wheat sowing window for Bathinda is 25 October to 15 "
        "November. Earlier risks heat stress at germination; later than 20 "
        "November risks grain-fill colliding with March-April heat waves. The "
        "1-10 November window typically balances both risks. Source: PAU "
        "Package of Practices, ICAR-IIWBR Karnal advisories.",
        "pau:wheat:sowing",
    ),
    (
        "My wheat is showing yellowing leaves at tillering. What should I do?",
        "",
        "Yellowing at tillering is most commonly nitrogen deficiency (older "
        "lower leaves) or sulfur deficiency (younger upper leaves). Apply 12 kg "
        "N per acre as urea topdressing for nitrogen, OR 8 kg sulfur per acre "
        "via gypsum or ammonium sulfate if upper leaves yellow. Avoid nitrogen "
        "in fog or cloudy weather. Also check for waterlogging. Source: PAU "
        "diagnosis guide, ICAR-IIWBR yellowing advisory.",
        "pau:wheat:diagnosis",
    ),
    (
        "What wheat variety is best for terminal heat tolerance in Punjab?",
        "",
        "For terminal heat tolerance: HD-3086, HD-3226, PBW-725, PBW-Zinc-1. "
        "HD-3086 and HD-3226 from ICAR-IIWBR show 8-12% yield retention under "
        "late sowing versus older varieties like PBW-343 which lose 20-25%. "
        "PBW-Zinc-1 also has biofortified zinc. Source: ICAR-IIWBR variety "
        "release notes 2018-2024.",
        "icar:wheat:varieties",
    ),
    (
        "How much irrigation does Punjab wheat need and when?",
        "",
        "Punjab wheat needs 4-6 irrigations totaling 35-40 cm. Critical timings: "
        "Crown root initiation (CRI) at 21 DAS, tillering at 45 DAS, jointing at "
        "65 DAS, booting at 85 DAS, milking at 105 DAS, dough at 120 DAS. The "
        "CRI irrigation is the single most yield-critical (missing it costs "
        "15-20% yield). With declining groundwater, drip or alternate-row "
        "irrigation cuts use 30%. Source: PAU Package of Practices.",
        "pau:wheat:irrigation",
    ),
]

RICE_EXAMPLES: list[tuple[str, str, str, str]] = [
    (
        "When can I transplant rice in Punjab this year?",
        "",
        "Punjab law mandates rice transplanting after 10 June (Subsoil Water "
        "Preservation Act). Recommended: 15-25 June for Basmati, 20 June - 5 "
        "July for non-Basmati. Earlier transplanting accelerates groundwater "
        "depletion. Direct-seeded rice (DSR) sown 1-10 June is the "
        "groundwater-friendly alternative. Source: Punjab government, PAU "
        "Package of Practices.",
        "pau:rice:sowing",
    ),
]

COTTON_EXAMPLES: list[tuple[str, str, str, str]] = [
    (
        "Should I grow cotton in Bathinda given climate change?",
        "",
        "Bathinda cotton is currently optimal (boll formation Aug-Sep "
        "averaging ~32°C). Projections show this rising to ~34°C by 2050, "
        "pushing cotton into marginal viability. Heat above 35°C during boll "
        "formation causes shedding; sustained marginal conditions are "
        "projected from around 2029. Options: heat-tolerant Bt hybrids "
        "(RCH-650 tolerates 36-38°C), shift planting to April for earlier "
        "boll formation, or diversify with guar/pulses. Source: ICAR-CRIDA "
        "climate atlas, FAO ECOCROP.",
        "crida:cotton:viability",
    ),
]

GENERAL_EXAMPLES: list[tuple[str, str, str, str]] = [
    (
        "El Niño is forecast this season. How does it affect Punjab wheat?",
        "",
        "El Niño typically reduces northwest India winter rainfall 15-25% and "
        "raises winter temperatures 0.5-1.0°C. For Punjab wheat: (1) reduce "
        "nitrogen 10% to avoid lodging from rapid growth, (2) advance sowing "
        "3-5 days to push grain-fill before March heat, (3) ensure two extra "
        "irrigations (CRI and tillering become more critical), (4) prefer "
        "heat-tolerant varieties like HD-3086. Source: ICAR-CRIDA ENSO "
        "advisories, IMD seasonal outlook.",
        "crida:enso:wheat",
    ),
    (
        "How do I improve organic carbon in Punjab alkaline soil?",
        "",
        "Punjab soils typically show OC below 0.5% (optimum 0.75%+). Best: "
        "(1) incorporate paddy residue with rotavator instead of burning "
        "(adds 0.05% OC per cycle), (2) green manure (dhaincha/sunhemp) in "
        "summer, (3) 5 tonnes FYM/compost per acre annually, (4) include "
        "legumes (mung/guar) in rotation. Expect 0.1% OC gain per 2-3 years "
        "of consistent practice. Source: PAU Soil Health Management.",
        "pau:soil:organic_carbon",
    ),
]


def build_dataset() -> list[dict]:
    """Assemble all examples into a unified instruction-tuning dataset."""
    all_examples = []
    for category, examples in [
        ("wheat", WHEAT_EXAMPLES),
        ("rice", RICE_EXAMPLES),
        ("cotton", COTTON_EXAMPLES),
        ("general", GENERAL_EXAMPLES),
    ]:
        for instruction, input_, output, source in examples:
            all_examples.append({
                "instruction": instruction,
                "input": input_,
                "output": output,
                "category": category,
                "source": source,
            })
    return all_examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    examples = build_dataset()
    output_path = Path(__file__).parent / "data" / "punjab_agronomy.jsonl"
    write_jsonl(examples, output_path)
    print(f"✓ Wrote {len(examples)} examples to {output_path}")
    print(f"  Wheat:   {sum(1 for e in examples if e['category'] == 'wheat')}")
    print(f"  Rice:    {sum(1 for e in examples if e['category'] == 'rice')}")
    print(f"  Cotton:  {sum(1 for e in examples if e['category'] == 'cotton')}")
    print(f"  General: {sum(1 for e in examples if e['category'] == 'general')}")
