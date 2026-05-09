"""Append climate-adaptation Q&A pairs to the agronomy dataset.

These tie PAU agronomy to: climate change, ENSO advisories, demo persona Ramesh
in Bathinda, the ClimateCalendar app's viability projection capability.

Run: python3 finetune/append_climate.py
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "punjab_agronomy.jsonl"


def Q(
    instruction: str,
    output: str,
    category: str = "general",
    source: str = "climate_calendar:v1",
) -> dict:
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": category,
        "source": source,
    }


ENTRIES = [
    # ===== Climate change in Punjab — observed and projected =====
    Q(
        "How is the climate changing in Punjab?",
        "Punjab's climate has warmed gradually over the past three decades. Average annual temperature has risen modestly, but rainfall has dropped sharply in many districts - Bathinda, for example, saw a 28 percent decline in rainfall between the 1995-1999 and 2020-2024 averages (640mm down to 458mm). CMIP6 climate model projections show a further +1.22 degrees C warming for Punjab between 2025 and 2050. The combination of warming temperatures and drier rainfall makes water-management and crop-choice decisions increasingly important.",
    ),
    Q(
        "What does CMIP6 say about Punjab's climate by 2050?",
        "CMIP6 (Coupled Model Intercomparison Project Phase 6) climate projections show Punjab warming by about +1.22 degrees C between 2025 and 2050. Annual mean temperature in Bathinda is projected to rise from 25.63 degrees C in 2025 to 26.85 degrees C in 2050. Combined with declining monsoon reliability, this warming pushes some crops (cotton in particular) from optimal toward marginal viability, while wheat and rice stay viable but with stress. ClimateCalendar uses these projections to estimate when a farmer's preferred crop may become unviable.",
    ),
    Q(
        "What is the climate fingerprint for Bathinda Punjab?",
        "Bathinda's climate fingerprint (1995-2024): mean annual temperature roughly 24.4 degrees C (essentially unchanged across recent 5-year windows). Annual rainfall: 1995-1999 average 640mm, 2020-2024 average 458mm - a 28 percent decline. May-June max temperatures regularly exceed 40 degrees C. CMIP6 projects +1.22 degrees C warming by 2050. Soil sample (ISRIC SoilGrids): loam, sand 50 percent, clay 22 percent, pH 7.8, soil organic carbon 8.5 g/kg (low), CEC 11. This profile is typical for the Malwa belt of Punjab.",
    ),
    Q(
        "How does declining rainfall in Punjab affect farming decisions?",
        "Punjab's monsoon rainfall has been declining - Bathinda's recorded 28 percent drop between 1995-1999 and 2020-2024 windows is representative of much of the Malwa belt. Lower rainfall means more reliance on tubewell irrigation, which depletes groundwater further. Practical adjustments: switch to short-duration rice varieties (PR 126, PR 130) to shorten the flooded period, adopt direct-seeded rice with tar-wattar method (15-20 percent water savings), use laser land levelling and tensiometer-based irrigation, prefer wheat residue retention (Happy Seeder, Super Seeder) to conserve soil moisture, and consider crop diversification away from rice on water-stressed fields.",
    ),
    # ===== ENSO and seasonal forecasts =====
    Q(
        "What is ENSO and why does it matter for my farm?",
        "ENSO is the El Nino-Southern Oscillation, a periodic shift in Pacific Ocean temperatures that strongly influences global rainfall patterns. NOAA tracks ENSO via the Oceanic Nino Index (ONI) - a 3-month average sea surface temperature anomaly. ENSO has three states: El Nino (warm phase, often weakens Indian monsoon and reduces Punjab rainfall), La Nina (cool phase, often strengthens monsoon and brings more Punjab rain), and Neutral (between the two). Knowing the ENSO state ahead of the kharif sowing season helps you decide on variety duration, irrigation budget, and risk management.",
    ),
    Q(
        "What is the current ENSO state and what does it mean for Punjab?",
        "As of February 2026, NOAA's Oceanic Nino Index reads -0.16 degrees C - a Neutral state, slightly on the cool side. Neutral years generally bring near-normal monsoon rainfall to Punjab, without the suppression of an El Nino or the surplus of a La Nina. For Ramesh in Bathinda, this means baseline planning: standard short-duration variety, full irrigation budget, no special preparation for either drought or flood. Continue to monitor NOAA updates as the season approaches - ENSO can shift by July.",
    ),
    Q(
        "Should I change my crop variety based on ENSO forecasts?",
        "Yes, strategically. In El Nino years (warm Pacific phase, suppressed monsoon), prioritize PR 126 or PR 130 short-duration rice that finish before water stress peaks; expand direct-seeded rice acreage; budget for extra irrigation; consider switching some rice acreage to alternative kharif crops like cotton or maize on lighter soils. In La Nina years (cool Pacific phase, strong monsoon), longer-duration varieties like PR 122 (117 days) become viable; ensure drainage on heavier soils; watch for waterlogging-related diseases like sheath blight and bacterial blight in rice. In Neutral years, follow standard PAU recommendations.",
    ),
    Q(
        "Where can I check the latest ENSO forecast?",
        "NOAA's Climate Prediction Center publishes monthly ENSO Diagnostic Discussions and the Oceanic Nino Index at https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php. The Indian Meteorological Department also publishes monsoon forecasts that incorporate ENSO state. ClimateCalendar can fetch the current ENSO status when asked. Re-check around April-May before kharif decisions, since ENSO phase can evolve.",
    ),
    # ===== Demo persona — Ramesh, Bathinda, wheat farmer =====
    Q(
        "I'm a wheat farmer in Bathinda Punjab. What is my biggest climate risk?",
        "Your biggest climate risk in Bathinda is March-April heat waves at grain filling stage. Bathinda's grain-fill window (mid-February to mid-April) increasingly sees temperatures spiking above 28 degrees C - the threshold past which wheat grain weight starts to shrink. Combined with the documented 28 percent decline in rainfall over the past 25 years, late-season moisture stress can compound the heat problem. Practical mitigations: prefer heat-tolerant varieties, sow on time (do not delay past mid-November) so grain-fill occurs in cooler weeks, use PAU-recommended potassium nitrate or salicylic acid spray for high-temperature stress, and ensure the third irrigation reaches the field before the heat peaks.",
    ),
    Q(
        "I farm in Bathinda Punjab. What is the climate outlook for my farm by 2050?",
        "Bathinda is projected to warm by about +1.22 degrees C between 2025 and 2050 per CMIP6, with annual mean temperature rising from 25.63 degrees C to 26.85 degrees C. Wheat will remain viable but face increasing stress at grain-fill (current grain-fill mean ~27.3 degrees C, projected ~29.3 degrees C by 2050 - approaching the upper PAU-tolerable range). Rice is already at the margin in Punjab today, not just future - the issue is current. Cotton in your area is currently optimal but will become marginal by 2050, with the threshold breach projected around 2029 - just 3 growing seasons from now. Plan crop diversification accordingly.",
    ),
    Q(
        "I'm worried about cotton viability in Bathinda. Should I keep growing it?",
        "Honest assessment: cotton in Bathinda is optimal NOW (current temperatures hit the 27-32 degrees C cotton sweet spot), but viability is projected to shift to marginal by 2050, with the threshold breach expected around 2029 - meaning you have roughly 3 more growing seasons before climate-driven stress becomes routine. PAU-recommended Bt varieties (PAU Bt 2, PAU Bt 3) and the standard agronomy will continue to perform reasonably until then. Plan for diversification: explore short-duration kharif alternatives (maize, soybean, or basmati rice depending on water availability), or adopt drip irrigation to reduce climate vulnerability of your cotton acreage. ClimateCalendar's viability_2050 tool can model your specific GPS location's trajectory.",
    ),
    Q(
        "I am Ramesh from Bathinda, a smallholder wheat farmer. What planting calendar should I follow this year?",
        "For Bathinda smallholder wheat planning: sow in the first half of November (timely sowing window) using a PAU-recommended variety like PBW 826 or DBW 187. Apply 50 kg N, 25 kg P2O5 per acre on medium fertile soil; treat seed against termite, smut, and flag smut; use Happy Seeder or Super Seeder if previous crop was rice (no straw burning). With current Neutral ENSO state and projected normal rainfall, follow the standard PAU schedule. Use PAU-LCC at the 50-55 day mark for need-based urea (often saves 30 kg urea per acre). Plan three irrigations: at CRI (21 days), tillering (~45 days), flowering, and grain-fill. Watch for March-April heat waves - have potassium nitrate spray ready if temperatures exceed 28 degrees C at grain-fill.",
    ),
    # ===== Climate-adapted variety choices =====
    Q(
        "Which wheat varieties are most heat-tolerant for Punjab's warming climate?",
        "PAU recommends several wheat varieties suited to current Punjab conditions. For heat tolerance specifically: HD 3226 and DBW 187 are widely grown for their performance under late-sown and warmer-end-of-season stress; PBW 826 and PBW 803 are also robust under typical Punjab conditions. As climate continues to warm, prefer varieties with shorter grain-fill periods (faster from flowering to maturity) so grain-fill finishes before peak heat. Combine with timely sowing (first half of November) and PAU's recommended potassium nitrate or salicylic acid spray for heat-stress mitigation.",
    ),
    Q(
        "How should I adapt rice variety choice if the monsoon weakens?",
        "Under weakening or delayed monsoon (often associated with El Nino years), prioritize short-duration rice varieties: PR 126 (93 days, broadest sowing window), PR 130 (105 days), or PR 121 (110 days, lodging-tolerant). All are PAU-recommended and resistant to multiple bacterial blight pathotypes. Combined with direct-seeded rice (DSR) in tar-wattar fields from 1 June onwards, you save 15-20 percent irrigation water versus puddled transplanted rice. This shortens the flooded period and gets you out of the field earlier so wheat can be sown on time.",
    ),
    Q(
        "Should I switch to drought-tolerant crops as Punjab warms?",
        "Punjab's projected +1.22 degrees C warming by 2050 doesn't require abandoning wheat or even rice on most fields - both stay viable per CMIP6 projections - but it does favour: (1) Short-duration varieties (PR 126 rice, faster wheat varieties) to shorten heat exposure. (2) Direct-seeded rice over puddled rice for water savings. (3) Diversification of some acreage into less water-intensive kharif options (maize, cotton on appropriate soils, pulses). (4) Long-term consideration of crop substitution where viability_2050 projections show breaches before 2050 (e.g., cotton in some Bathinda fields by 2029). PAU's published variety lists are updated annually with climate-relevant new releases.",
    ),
    # ===== Adaptation practices that connect agronomy to climate =====
    Q(
        "How does Happy Seeder help with climate change?",
        "Happy Seeder is doubly useful for climate adaptation. (1) It eliminates paddy straw burning, which is a major contributor to seasonal air pollution AND releases CO2, methane, and nitric oxide. One tonne of burnt straw releases 400 kg of carbon - retention via Happy Seeder keeps that carbon in soil, building organic matter that improves drought resilience. (2) The straw mulch left after Happy Seeder sowing reduces soil evaporation, helping wheat conserve moisture during increasingly dry Punjab winters. After 8 years of continuous use, soil organic carbon builds enough that you can also save 20 kg urea per acre. Win-win for farmer cost AND climate.",
    ),
    Q(
        "How does direct-seeded rice (DSR) help with climate adaptation?",
        "DSR (direct-seeded rice in tar-wattar fields from 1 June) saves 15-20 percent irrigation water compared to puddled transplanted rice. As Punjab's groundwater declines and monsoon reliability weakens, this water saving becomes increasingly valuable. DSR also: (1) Reduces methane emissions from the rice paddy (puddled rice fields are major CH4 sources). (2) Shortens labour requirement at peak transplanting time. (3) Frees up the field earlier for timely wheat sowing. Combined with short-duration varieties, DSR is one of the highest-impact climate adaptation strategies for Punjab rice.",
    ),
    Q(
        "How does laser land levelling help in a changing climate?",
        "Laser land levelling improves water distribution uniformity, reducing total irrigation water use significantly per acre. In a climate where rainfall is declining and groundwater levels falling, this water efficiency directly extends your irrigation budget. Laser-levelled fields also allow precision irrigation timing with tensiometers, further saving water without yield loss. PAU recommends laser levelling before direct sowing or transplanting of rice. The investment pays back in 2-3 seasons through water and labour savings.",
    ),
    Q(
        "Why is paddy straw retention important for climate?",
        "Paddy straw retention (via Happy Seeder, Super Seeder, Smart Seeder, or surface seeding) instead of burning has three climate benefits. (1) Reduced air pollution and greenhouse gas emissions - one tonne of burnt straw releases CO2, CO, methane, and nitric oxide, all harmful. (2) Soil carbon sequestration - the 400 kg of carbon per tonne stays in the soil instead of going to the atmosphere. (3) Improved drought resilience - higher soil organic matter means better water retention during increasingly dry winters. PAU's recommendation to retain straw is one of the highest-leverage climate-positive practices available to Punjab farmers today.",
    ),
    # ===== Heat stress and grain-fill =====
    Q(
        "What temperature threshold harms wheat grain filling?",
        "Wheat grain weight starts to shrink when temperatures exceed roughly 28 degrees C at the grain-fill stage. In Punjab, this stage falls in March-April when heat waves are increasingly common. The current Bathinda grain-fill mean is around 27.3 degrees C - already at the threshold. CMIP6 projects this rising to about 29.3 degrees C by 2050, putting wheat under increasing stress. Mitigations: timely sowing in the first half of November so grain-fill happens in cooler weeks; PAU-recommended potassium nitrate or salicylic acid spray; choose varieties with shorter grain-fill duration so the heat-vulnerable window is brief.",
    ),
    Q(
        "What can I spray on my wheat to protect it from late-season heat?",
        "PAU recommends two options for high-temperature stress at the grain-filling stage: (1) Potassium nitrate spray at the recommended rate (typically 1-2 percent solution). (2) Salicylic acid spray at the recommended concentration. Both help maintain grain-fill metabolism under heat stress and protect grain weight. Apply when temperatures exceed 28 degrees C during grain-fill (typically late March or April in Punjab). Combined with timely sowing and a heat-tolerant variety, sprays can substantially reduce yield loss in hot years.",
    ),
    Q(
        "Why is March-April heat dangerous for Punjab wheat?",
        "March-April is wheat's grain-fill stage in Punjab - the period when individual grain weight is determined. Temperatures above 28 degrees C during this window accelerate ripening and shrink grain size, reducing yield even if everything else is perfect. Punjab's March-April mean temperatures have been creeping up, and CMIP6 projects the trend continuing. This is why sowing wheat in the first half of November (the timely sowing window) matters so much - it ensures grain-fill completes in cooler February-March weather rather than late March-April when heat risk peaks.",
    ),
    # ===== Soil health and organic carbon =====
    Q(
        "Why does soil organic carbon matter for climate adaptation?",
        "Soil organic carbon (SOC) does three things relevant to climate. (1) It improves water retention, making the soil more drought-resilient as Punjab's rainfall declines. (2) It improves nutrient mineralization, reducing chemical fertilizer requirements - PAU specifically allows 20 kg urea per acre savings on rice (and similar on wheat) once SOC reaches the 'high' category. (3) It sequesters carbon - keeping atmospheric CO2 in the soil. Bathinda's typical SOC reading is 8.5 g/kg, which is in the LOW category. Building this through paddy straw retention, FYM, prali char, and green manure is a high-leverage long-term investment.",
    ),
    Q(
        "My soil organic carbon is low. How do I build it up?",
        "Five PAU-recommended practices to build soil organic carbon over time: (1) Continuous paddy straw retention via Happy Seeder, Super Seeder, or surface seeding - no burning. (2) Apply 6 tonnes per acre of farmyard manure, or 6 tonnes pressmud, or 2.5 tonnes poultry manure before transplanting/sowing. (3) Apply 2.0 tonnes per acre of prali char (paddy straw biochar) - boosts yield 10 percent on top of carbon benefits. (4) Green manuring with dhaincha, cowpea, or sunnhemp before kharif crops. (5) Summer moong residue incorporation. After about 8 years of continuous practice, SOC reaches the 'high' category and you can save 20 kg urea per acre per crop - real financial value alongside climate benefit.",
    ),
    Q(
        "What is prali char and how does it help climate adaptation?",
        "Prali char is biochar made from paddy straw via partial low-oxygen combustion in a clay-and-brick dome kiln (12 quintals straw -> 8 quintals char). Apply 2.0 tonnes per acre to wheat or rice. Climate benefits: (1) Carbon sequestration - the porous biochar locks 30-36 percent carbon in stable form for decades to centuries. (2) Productivity gain - boosts crop yield 10 percent and saves 16 kg N (35 kg urea) per acre. (3) Soil health - improves water retention and microbial habitat. (4) Solves paddy straw burning while turning waste into a valuable soil amendment. PAU provides a kiln design (14 ft tall, 10 ft diameter) for on-farm production.",
    ),
    # ===== Water management =====
    Q(
        "My groundwater is declining. How can I save irrigation water?",
        "Multiple PAU-recommended practices stack for water savings. (1) Switch to short-duration rice varieties (PR 126 at 93 days, PR 130 at 105 days). (2) Adopt direct-seeded rice in tar-wattar fields - saves 15-20 percent water versus puddled transplanted rice. (3) Use laser land levelling for uniform irrigation. (4) Use tensiometer-based irrigation - irrigate only when tension reaches 150 plus or minus 20 cm at 15-20 cm depth, or when water level enters the yellow strip. (5) Switch from continuous flooding to intermittent (irrigate 2 days after ponded water has infiltrated). (6) On heavy soils, consider ridge or bed transplanting which saves significant water versus flat puddled. Each of these alone saves 5-15 percent; combined they can cut total water use 40 percent or more.",
    ),
    Q(
        "How does intermittent flooding save water in rice?",
        "Continuous flooding wastes water that percolates below the root zone or evaporates from the standing water surface. PAU's intermittent flooding approach - keep water standing only for the first 2 weeks (establishment), then irrigate only 2 days AFTER the previously ponded water has infiltrated - cuts irrigation cycles substantially without yield loss. Combined with a tensiometer at 15-20 cm depth (irrigate when matric tension reaches 150 plus or minus 20 cm), the savings can reach 30-40 percent of total water use. Take care that the field does not develop cracks - they cause large deep-percolation losses.",
    ),
    Q(
        "Why is tensiometer-based irrigation good for climate adaptation?",
        "A tensiometer measures soil moisture tension precisely, telling you when the rice plants actually need water rather than relying on calendar-based or visual irrigation. Install at 15-20 cm depth. Irrigate when matric tension reaches 150 plus or minus 20 cm, OR when water level in the tensiometer enters the yellow strip. This precision irrigation: (1) Saves 20-30 percent water versus calendar-based scheduling. (2) Avoids both over-irrigation (which wastes water and increases pest/disease pressure) and under-irrigation (which causes yield loss). (3) Adapts automatically to rainfall - if a storm comes, the tensiometer tells you to skip the next irrigation. As Punjab's groundwater declines, this tool earns back its small cost in one season.",
    ),
    # ===== Extension and decision-support =====
    Q(
        "Where can I get reliable PAU agronomy advice?",
        "Punjab Agricultural University (PAU) publishes updated Package of Practices for kharif (May) and rabi (October) every year, available through the PAU bookshop in Ludhiana, Krishi Vigyan Kendras (KVKs) in every district, and Farm Advisory Service Centres. The PAU-LCC, biofertilizers, recommended seeds, and other inputs are also available through these channels. Online: pau.edu and the PAU-Urea Guide app for need-based nitrogen management. ClimateCalendar combines PAU's recommendations with climate projections, ENSO state, and your specific location's soil profile to provide personalized, climate-aware planting advice.",
    ),
    Q(
        "What is the PAU Package of Practices?",
        "PAU's Package of Practices is the authoritative agronomic reference for Punjab farmers, published twice a year - kharif edition in May (covering rice, cotton, maize, pulses, etc.) and rabi edition in October (wheat, mustard, gram, etc.). It contains: variety recommendations with disease-resistance ratings, sowing windows, seed rates, fertilizer recommendations, irrigation schedules, weed management, pest and disease management with specific brand-name insecticides and doses, harvesting and storage guidance. Updated annually with new variety releases and current best practices. ClimateCalendar's tools and recommendations are grounded in this material.",
    ),
    Q(
        "How is ClimateCalendar different from regular agronomy advice?",
        "ClimateCalendar combines authoritative PAU agronomy guidance with climate-aware tools that regular extension services don't typically provide: (1) Real-time climate trend analysis using Open-Meteo's ERA5 reanalysis data. (2) Projections through 2050 using CMIP6 climate models. (3) ENSO state lookup using NOAA's Oceanic Nino Index. (4) Soil profile retrieval from ISRIC SoilGrids based on your GPS coordinates. (5) Crop viability projection through 2050 with breach-year estimates. (6) Multilingual, multimodal interface (text, photo, voice) so smallholder farmers can use it offline on a basic smartphone. The result: PAU's agronomy advice plus the climate context to make decisions that hold up over the coming decades.",
    ),
    # ===== Misc cross-cutting questions =====
    Q(
        "Why is climate adaptation urgent for Punjab farmers?",
        "Three reasons. (1) The water crisis is now: groundwater is declining 0.5-1 metre per year in many districts, and rainfall has dropped substantially in the past 25 years. (2) Some climate breaches are imminent: ClimateCalendar projects cotton in Bathinda to shift from optimal to marginal viability around 2029 - just 3 growing seasons from May 2026. (3) Compound stress: rising temperatures during wheat grain-fill (already at the 28 degrees C threshold), weakening monsoons under climate change, and groundwater depletion stack to make the next decade harder than the past three. Farmers who adapt early - via short-duration varieties, direct seeding, tensiometer irrigation, paddy straw retention, soil organic carbon building - will be substantially better positioned than those who wait.",
    ),
    Q(
        "What single thing should every Punjab farmer do for climate resilience?",
        "Stop burning paddy straw. This single practice change addresses multiple climate problems at once: (1) Eliminates massive air pollution and greenhouse gas emissions (CO2, methane, CO, nitric oxide). (2) Retains 400 kg carbon, 5.5 kg N, 2.3 kg P, 25 kg K, 1.2 kg S per tonne of straw in the soil instead of the atmosphere. (3) Builds soil organic matter for long-term drought resilience. (4) Allows direct wheat sowing with Happy Seeder, Super Seeder, or surface seeding - saves time, labour, and an irrigation. (5) After 8 years of continuous retention, soil organic carbon reaches the 'high' category and PAU allows saving 20 kg urea per acre on rice (and similar on wheat). PAU has provided multiple machines, methods, and ex-situ uses (biogas, prali char, biomass power) - the technology exists. The behavior change is the bottleneck.",
    ),
    Q(
        "If I have to pick one variety to grow this year, what should I grow as a smallholder in Bathinda?",
        "For wheat (rabi season): PBW 826 or DBW 187 - both PAU-recommended, productive on Bathinda's loam soils, with good disease resistance and reasonable heat tolerance for the increasingly warm grain-fill season. Sow in the first half of November (timely window). For rice (kharif season): PR 126 (93 days, water-saving, full bacterial blight resistance, broad sowing window) or PR 130 (105 days, lodging-tolerant, full bacterial blight resistance) - both designed for water-scarce Punjab conditions. With Bathinda's projected climate trajectory (warming +1.22 degrees C by 2050, declining rainfall), short-duration varieties are the safer choice.",
    ),
    Q(
        "How can I monitor my farm's climate risk over time?",
        "Track three metrics annually: (1) Cumulative rainfall during the kharif season (June-September) and rabi season (November-March) - compare to your 5-year and 10-year averages to spot trends. (2) Maximum daily temperature during wheat grain-fill (mid-February to mid-April) - count days above 28 degrees C, the wheat heat-stress threshold. (3) Tubewell water level (depth to water table) before and after each kharif season - rising depth indicates groundwater drawdown. ClimateCalendar can pull this data from public sources for your specific GPS location, including comparing your local trends to district and state averages. The annual review helps you spot when your farm-specific risk is changing faster than the regional average.",
    ),
    Q(
        "What does it mean when ClimateCalendar says my crop has 'breach year 2029'?",
        "The 'breach year' is when ClimateCalendar's viability projection model expects the climate at your specific GPS location to cross a key threshold for that crop. For Bathinda cotton, the projected breach year is 2029 - meaning by then, current optimal cotton conditions (27-32 degrees C day temperatures during fruiting, cool nights, dry sunny October-November pickings) will have shifted enough that yield variability and stress risk increase substantially. The crop is still growable - just less reliable. Plan for diversification before the breach year arrives. The model uses CMIP6 climate projections combined with PAU's documented thermal tolerance ranges to estimate the breach year for your fields.",
    ),
    Q(
        "My family has grown rice for generations. Should I stop?",
        "Punjab's rice problem is current, not future - your fields have been getting harder to irrigate every year as groundwater declines. The PAU recommendations have been adjusting in response: short-duration varieties, direct seeding, intermittent flooding, tensiometer-based irrigation - all are designed to maintain rice farming with much less water. You don't need to stop, but you do need to change methods. ClimateCalendar's analysis of your specific location can tell you whether continued rice farming is viable on your fields with adjusted methods, or whether partial diversification (some acreage to maize, cotton, or pulses) is wise. The honest answer varies by location - your choice should be data-driven, not based on fear.",
    ),
    Q(
        "Should I trust government MSP for my farming planning?",
        "Minimum Support Price (MSP) is announced annually by the Government of India for major crops and provides a price floor for procurement (mainly wheat and paddy in Punjab). Plan your annual cropping based on it - it gives you a reasonable income guarantee. However, MSP-driven incentives have contributed to Punjab's rice-wheat mono-cropping problem and groundwater depletion. For climate-resilient long-term planning, balance MSP-anchored choices (wheat, paddy) with diversification crops (maize, cotton, pulses, oilseeds) that may not have MSP but have better water-economics under climate change. Also follow PAU's variety recommendations to maximize yield under the chosen crop.",
    ),
    Q(
        "What is the most important thing a young farmer in Punjab should learn today?",
        "Three things, in order of impact. (1) Soil testing - apply fertilizer based on your specific soil's needs, not a generic recommendation. PAU has a dedicated Soil Testing chapter and KVKs across Punjab support this. Saves money and prevents over-application. (2) Need-based nitrogen via PAU-LCC or Green Seeker - a 200 rupee colour chart can save 30 kg urea per acre per season; that pays for itself many times over. (3) Climate-aware crop choice - know what your specific GPS location's climate trajectory is, and pick varieties and crops that match where the climate is heading, not just where it was. ClimateCalendar plus PAU's Package of Practices, used together, give a young farmer all the information needed to be more productive AND more sustainable than the previous generation.",
    ),
]


def main() -> None:
    existing = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["instruction"])
                except json.JSONDecodeError:
                    pass

    new_entries = [e for e in ENTRIES if e["instruction"] not in existing]
    skipped = len(ENTRIES) - len(new_entries)

    with OUT.open("a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"Appended {len(new_entries)} climate entries (skipped {skipped} duplicates)")
    print("Dataset now contains:")
    subprocess.run(["wc", "-l", str(OUT)], check=False)


if __name__ == "__main__":
    main()
