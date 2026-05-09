"""
Append PAU cotton Q&A pairs to the agronomy dataset.
Run: python3 finetune/append_cotton.py
"""
import json
import subprocess
from pathlib import Path

CATEGORY = "cotton"
SOURCE = "pau:cotton:kharif_2025"
SUFFIX = "Source: PAU Package of Practices Kharif 2025."

def Q(instruction, output):
    return {
        "instruction": instruction,
        "input": "",
        "output": output + " " + SUFFIX,
        "category": CATEGORY,
        "source": SOURCE,
    }

ENTRIES = [
    # ===== Chunk 02: Sub-soiling, sowing, seed rate, acid delinting =====
    Q("What is sub-soiling and why does PAU recommend it for cotton?",
      "Cross sub-soiling at 1.0 metre spacing should be done before preparing the cotton field. Use a tractor-drawn sub-soiler (chiseler) to a depth of 45-50 cm. Then plank to break clods and prepare a fine seedbed. Sub-soiling breaks the hard pan, increases water infiltration rate, and improves root development of cotton plants - critical for cotton's deep root system."),

    Q("How should I prepare the field for sowing cotton?",
      "A fine seed-bed is essential for securing a good plant stand. During field preparation before sowing cotton, incorporate any wheat straw left from combine harvesting into the soil using a rotavator or super seeder. Combine sub-soiling at 1.0 m spacing with the chiseler at 45-50 cm depth, then plank to break clods, then make a fine seedbed."),

    Q("What is the cotton sowing window in Punjab?",
      "Sow cotton during 1 April to 15 May. Sowing within this period ensures better yield and helps the crop escape attack from insect pests and diseases that build up later in the season. Do the actual sowing in morning and evening hours to avoid mid-day heat stress on emerging seedlings."),

    Q("How much PAU Bt 2 or PAU Bt 3 seed do I need per acre?",
      "Use 4.0 kg of PAU Bt 2 or PAU Bt 3 cotton seed per acre, plus 1.0 kg of non-Bt refuge seed grown around the Bt cotton. The refuge prevents bollworms from evolving resistance to the Bt toxin. Total seed: 5.0 kg/acre."),

    Q("How much Bt cotton hybrid seed do I need per acre?",
      "Use 0.900 kg of recommended Bt cotton hybrid seed per acre, plus 0.240 kg of non-Bt refuge seed - OR use two pouches of 475 grams each that already contain mixed refuge. The refuge prevents bollworms from evolving resistance to the Bt toxin."),

    Q("How much non-Bt cotton seed do I need per acre?",
      "Use 3.5 kg of non-Bt cotton seed per acre (F 2228 or LH 2108 varieties). For desi cotton varieties, use 3.0 kg per acre."),

    Q("Why does PAU recommend a refuge of non-Bt cotton around Bt cotton?",
      "The non-Bt refuge prevents bollworms from evolving resistance to the Bt toxin. If only Bt cotton is grown, surviving resistant bollworms can multiply quickly. The refuge keeps a population of susceptible bollworms that mate with any resistant survivors, diluting resistance genes. PAU recommends 20 percent area as non-Bt cotton hybrids around Bt cotton, OR 5 percent unsprayed non-Bt area. Use the same variety's non-Bt version, or fall back on F 2228 or LH 2108."),

    Q("How do I acid-delint cotton seed at home?",
      "Mix 100 g commercial-grade concentrated sulphuric acid with 1 kg cotton seed in an earthen or plastic container. Stir vigorously for 2-3 minutes with a thick wooden stick. As soon as the fuzz dissolves, add 10 litres of water, stir well, and drain through a perforated plastic basket. Repeat washing 3 times. Then dip the seed for 1 minute in sodium bicarbonate solution (12.5 g in 2.5 litres water) to neutralize residual acid. Final water wash, remove floating damaged seed, dry in shade."),

    Q("What precautions should I take during cotton seed acid delinting?",
      "Four critical precautions: (1) Do not use metal or wood containers - acid will react with them. (2) The operator must wear plastic gloves. (3) Dispose of acid-and-alkali wash water properly in waste land, not into channels. (4) Inadequate or delayed washing or unneutralized residual acid will impair seed germination. Always follow with the sodium bicarbonate dip and final water rinse."),

    Q("Is there a non-acid alternative for delinting cotton seed?",
      "Yes - rub the non-delinted seed with fine earth, cow-dung, or ash to remove its fuzz and ensure uniform sowing. This is safer than acid delinting and works for farmers without protective equipment, though acid delinting also kills pink bollworm larvae and is preferred when feasible."),

    Q("What is cotton seed priming and how do I do it?",
      "Seed priming promotes good plant stand establishment, better early growth, and higher yield. Soak the seed in a solution of 0.5 g succinic acid in 5 litres of water - for 2-4 hours if the seed is acid-delinted, or 6-8 hours if non-delinted. Then drain and sow."),

    Q("What is cotton seed bio-priming?",
      "Dilute 100 mL of liquid bacterial inoculant of Bacillus thaonhiensis in 125-150 mL of water. Soak the recommended quantity of cotton seed for one acre in the diluted inoculant for 2 hours, then shade dry for half an hour before sowing. This biological priming improves productivity."),

    Q("How do I manage cotton seed for sodic water irrigation?",
      "In soils irrigated with sodic water (RSC greater than 2.5 meq per litre), treat seed with the liquid bioformulations Halo-Azo plus PSB plus ZnSB along with gypsum at 25 percent of the gypsum requirement. This reduces adverse effects of sodic water on cotton in the cotton-wheat system. The bioformulations are available at ICAR-CSSRI Regional Research Station, Lucknow at a nominal price."),

    Q("How do I treat cotton seed against jassid?",
      "Smear cotton seed with 5 g Gaucho 70 WS (imidacloprid) per kg seed, OR 7 g Cruiser 30 FS (thiomethoxam) per kg seed. This seed treatment prevents damage by cotton jassid during early establishment. Particularly important on jassid-susceptible cultivars."),

    Q("What row spacing should I use for cotton in Punjab?",
      "Sow cotton in lines 67.5 cm apart using a cotton sowing drill or cotton planter. The within-row plant spacing varies by type: non-Bt varieties at 60 cm, PAU Bt 2 and PAU Bt 3 at 30 cm, and Bt hybrids at 75 cm. Thin to the target spacing after first irrigation or heavy shower."),

    Q("What plant spacing should I use within rows for PAU Bt 2 and Bt 3 cotton?",
      "Plant-to-plant spacing within rows for PAU Bt 2 and PAU Bt 3 is 30 cm, after thinning. This is closer than non-Bt varieties (60 cm) or Bt hybrids (75 cm) because the PAU Bt varieties have a more compact growth habit. Row-to-row spacing is 67.5 cm for all types."),

    Q("What plant spacing should I use within rows for Bt cotton hybrids?",
      "For Bt cotton hybrids, plant-to-plant spacing within rows should be 75 cm after thinning. Row spacing is 67.5 cm. This wider plant spacing reflects the larger canopy of hybrids compared to varieties."),

    Q("Should I intercrop cotton with maize or cowpea?",
      "Yes - intercrop one row of maize or cowpea for fodder in cotton sown at 67.5 cm row spacing. This gives higher income compared to sole cotton. Apply recommended fertilizers to cotton and intercrops on an area basis. Harvest the maize/cowpea fodder at 45-55 days after sowing."),

    Q("Can I do in-situ green manuring with cotton?",
      "Yes - intercrop sunnhemp for in-situ green manuring. During cotton planting, sow two rows of sunnhemp at 22.5 cm spacing between the wider 67.5 cm cotton rows. Use 13 kg sunnhemp seed per acre. Incorporate the sunnhemp biomass at 35-40 days after sowing through mechanical interculture after the first irrigation. This improves cotton yield and soil physical properties."),

    Q("What is ridge sowing for cotton and why is it useful?",
      "Sowing cotton on ridges prepared with a cotton planter, and irrigating in furrows, saves considerable amount of irrigation water without reducing seed cotton yield. The crop is rainfed-friendly, root zone gets better aeration, and excess rainfall drains off the ridges. Recommended especially under water scarcity."),

    Q("Can I transplant cotton seedlings for gap filling?",
      "Yes - 3-week-old cotton nursery grown in 4x6 inch polythene bags filled with a 1:1 mixture of soil and FYM can be transplanted to fill gaps where direct-sown seed failed to establish. This restores plant population to recommended density."),

    # ===== Chunk 04: Weed control, fertilizer =====
    Q("How do I control itsit weed in cotton with herbicide?",
      "Apply 1.0 litre per acre of Stomp 30 EC (pendimethalin) as pre-emergence within 24 hours of sowing. Alternatively, Stomp 30 EC can be applied as post-emergence after first irrigation in 200 litres of water. Stomp controls itsit, madhana, and makra. Spray with tractor-mounted sprayer fitted with flat fan nozzle, in morning or evening hours."),

    Q("How do I apply pendimethalin herbicide for cotton?",
      "Apply 1.0 litre per acre of Stomp 30 EC (pendimethalin) as pre-emergence within 24 hours of cotton sowing. If weeds emerge after first irrigation or rain, apply Stomp 30 EC as post-emergence after the first irrigation, in 200 litres of water. Light hoeing before spray helps if some weeds already emerged."),

    Q("What is Hitweed Maxx and when do I use it in cotton?",
      "Hitweed Maxx 10 MEC contains pyrithiobac sodium 6 percent plus quizalofop ethyl 4 percent. Spray 500 mL per acre dissolved in 150 litres of water after first irrigation, in moist soil, to control annual grasses and broadleaf weeds. It also effectively controls lapeta (guara) and vel (Ipomoea sp.) when weed plants are at 2-5 leaf stage."),

    Q("Can I use paraquat for weed control in standing cotton?",
      "Yes - at 6-8 weeks after sowing when the crop is 40-45 cm tall, spray 500 mL Gramoxone 24 SL (paraquat) OR 900 mL Sweep Power 13.5 SL (glufosinate ammonium) per acre in 100 litres of water as a DIRECTED spray between crop rows using a protective hood. Both are non-selective and will damage cotton if they fall on crop leaves - the hood is essential."),

    Q("How much fertilizer does non-Bt cotton need on medium fertility soil?",
      "Apply 30 kg N and 12 kg P2O5 per acre on medium fertility soil. Sources: 65 kg urea and either 27 kg DAP or 75 kg single superphosphate. If wheat preceding the cotton received the recommended dose of phosphorus, omit phosphorus on cotton. If using 27 kg DAP, reduce urea by 10 kg."),

    Q("How much fertilizer does Bt cotton variety (PAU Bt 2 or Bt 3) need?",
      "Apply 37 kg N and 12 kg P2O5 per acre on medium fertility soil. Sources: 80 kg urea and either 27 kg DAP or 75 kg single superphosphate. If wheat preceding the cotton received the recommended phosphorus dose, omit phosphorus on cotton. If using 27 kg DAP, reduce urea by 10 kg."),

    Q("How much fertilizer does Bt cotton hybrid need?",
      "Apply 55 kg N and 12 kg P2O5 per acre on medium fertility soil. Sources: 120 kg urea and either 27 kg DAP or 75 kg single superphosphate. Bt hybrids need more nitrogen than varieties because of their larger biomass. If wheat preceded with full phosphorus, omit P on cotton. With 27 kg DAP, reduce urea by 10 kg."),

    Q("Should I apply potash to my cotton on light soil?",
      "Yes - on light soils, apply 20 kg muriate of potash per acre to cotton. Also apply 10 kg zinc sulphate heptahydrate (21 percent) OR 6.5 kg zinc sulphate monohydrate (33 percent) per acre. Light soils often show potassium and zinc deficiency that can limit cotton yield."),

    Q("When should I apply nitrogen splits to cotton?",
      "Apply half the nitrogen at thinning and the remaining half at the appearance of flowers. If the soil is low in fertility, apply the first half dose at sowing instead of at thinning. Drill all phosphorus at sowing along with 25 kg magnesium sulphate as basal dose."),

    Q("Should I apply boron to my cotton?",
      "Apply 400 g boron (4 kg borax) per acre AT SOWING only to boron-deficient (less than 0.5 kg available boron per acre) calcareous soils that have 2 percent or more calcium carbonate. Boron should NOT be applied indiscriminately - excessive boron causes toxicity. Test soil first."),

    Q("When does PAU say to skip phosphorus on cotton?",
      "Omit phosphorus on cotton when it follows wheat that received the recommended dose of phosphorus. Wheat is more phosphorus-responsive than cotton, and the residual phosphorus from the wheat application meets cotton's lower demand. This saves fertilizer cost without yield penalty."),

    # ===== Chunk 05: PAU-LCC, growth retardant, irrigation =====
    Q("How do I use PAU-LCC for cotton nitrogen management?",
      "Match leaf colour greenness of the topmost fully-developed intact leaf from 10 randomly-selected cotton plants with PAU-LCC under shade of your body, at thinning and at flower initiation. Apply urea based on the colour of 6 or more leaves out of 10: more than LCC shade 4.5 = 0 kg urea/acre; shade 4.5 = 20 kg; shade 4.0 = 35 kg; shade 3.5 or below = 50 kg."),

    Q("My cotton leaves are very dark green. How much urea should I apply?",
      "If 6 or more out of 10 cotton leaves match more than LCC shade 4.5 (very dark green), apply 0 kg urea per acre - skip the dose. The dark green colour shows the crop has enough nitrogen and adding more would worsen pest pressure (whitefly, bollworm) without yield gain. Use PAU-LCC for need-based nitrogen instead of fixed schedules."),

    Q("My cotton leaves are pale (LCC shade 3.5 or lower). What urea dose?",
      "If 6 or more out of 10 cotton leaves match LCC shade 3.5 or below (pale), apply 50 kg urea per acre at that timing. Pale leaves indicate nitrogen deficiency and the larger dose corrects it. Always check that the field is otherwise healthy (no pest, disease, water stress, other nutrient deficiency) before relying on LCC readings."),

    Q("What conditions must be met to use PAU-LCC accurately on cotton?",
      "Three conditions: (1) Leaves selected for measuring greenness should be free from insect or disease incidence. (2) The crop should not be under water stress or waterlogging. (3) Other nutrients (P, K, magnesium, zinc, boron) should be supplied per recommendations. If any of these are off, the LCC reading is unreliable."),

    Q("Where can I buy PAU-LCC for cotton?",
      "PAU-LCC is available at the PAU Seed Shop at Gate No. 1, Ludhiana, and through Krishi Vigyan Kendras and Farm Advisory Service Centres in different districts of Punjab. The same chart is used for cotton, rice, and wheat with different shade thresholds for each."),

    Q("What potassium nitrate sprays should I give my cotton?",
      "Give 4 sprays of 2 percent potassium nitrate (13:0:45) at weekly intervals starting at flower initiation. Dissolve potassium nitrate at 2 percent (about 2 kg in 100 litres of water per acre per spray). The sprays improve flower retention, boll set, and yield - especially valuable under late-season heat stress."),

    Q("How do I manage leaf reddening in Bt cotton?",
      "Give 2 sprays of 1 percent magnesium sulphate (1 kg magnesium sulphate in 100 litres of water per acre) at 15 days interval during full bloom and boll development stages. This corrects magnesium deficiency that causes leaf reddening, gives higher yield, and supports healthy boll filling."),

    Q("What should I do if my cotton is growing too tall vegetatively in heavy soil?",
      "Excessive vegetative growth in heavy soils blocks sunlight and causes shedding of buds, flowers, and bolls. Give 2 sprays of 300 mL Chamatkar (mepiquat chloride 5 percent w/w) per acre at 60 and 75 days after sowing, using 80-100 litres of water. This growth retardant checks excessive canopy growth without yield loss."),

    Q("How many irrigations does cotton need in a season?",
      "Cotton requires 4-6 irrigations depending on seasonal rainfall. The first irrigation should be 4-6 weeks after sowing; subsequent ones at 2-3 week intervals. On light soils or for crops sown on ridges, the first irrigation can be advanced if necessary. The LAST irrigation in September is critical to hasten boll opening."),

    Q("When is the most critical time for cotton water stress?",
      "Cotton must NOT be allowed to suffer water stress during the flowering and fruiting stages. Stress at this stage causes severe shedding of flowers and bolls, leading to large yield losses. Plan irrigations to keep the soil profile adequately moist through the August-September period."),

    Q("Cotton during early growth - how sensitive is it to water stagnation?",
      "Cotton during its early growth is VERY sensitive to water stagnation - it can die quickly. If water stands in the field after rains, drain it out immediately. This is one reason why ridge sowing or laser-levelled fields with proper drainage matter so much for cotton."),

    Q("What should I do if there is salinity in my irrigation water for cotton?",
      "In soils irrigated with saline water (EC up to 10 dS/m), apply 16 quintals per acre of rice-residue biochar - it reduces the adverse effect of salinity and increases seed cotton yield. Under poor-quality irrigation water in general, give pre-sowing irrigation with canal water and apply subsequent irrigations with poor-quality tubewell water in alternate furrows only."),

    # ===== Chunk 06: Salicylic acid, drip, IPM cultural =====
    Q("How do I save my cotton from sudden water stress?",
      "If cotton faces water stress due to no rainfall or sudden canal closure, dissolve 12.5 g of salicylic acid in 375 mL of ethyl alcohol, then add to 125 litres of water for spraying per acre on stress appearance. This minimizes yield loss from stress. CAUTION: do not apply under well-watered conditions - it will not increase yield and may cause stress symptoms."),

    Q("How should I drip irrigate American Bt cotton hybrids?",
      "Drip irrigate American Bt cotton hybrids at 7-day intervals. Lay laterals 67.5 cm apart with drippers 75 cm apart, dripper discharge 2.2 litres per hour. Irrigation time per cycle: 50 minutes in May/June, 45 minutes in July, 40 minutes in August, 35 minutes in September. If your dripper discharge differs, scale time proportionally with: Adjusted time = (2.2 x base time) / your dripper discharge."),

    Q("How do I do drip fertigation in cotton?",
      "Start fertigation of 100 kg urea (45 kg N) per acre at 35 days after sowing. Complete in 110-120 days using 10 equal splits at 7-day intervals. This delivers nitrogen precisely matching crop demand and dramatically improves nitrogen-use efficiency over broadcast urea."),

    Q("How can I irrigate cotton when good water is scarce?",
      "Under scarcity of good-quality irrigation water, alternate use of good-quality canal water and saline tubewell water through surface drip irrigation is recommended in light-textured soils. This gives sustainable seed cotton yield with minimal adverse effect on soil quality. For sub-surface drip details see PAU's Multiple Cropping chapter."),

    Q("What pest management warnings did PAU give for cotton?",
      "Six cultural/IPM principles: (1) Grow only recommended Bt cotton cultivars. (2) Prefer desi cotton in areas of high whitefly and leaf curl pressure. (3) Acid-delint or sun-dry seed-cotton 3-4 days in April to kill pink bollworm larvae. (4) Complete sowing by 15 May. (5) Avoid excessive nitrogenous fertilizer. (6) Eradicate alternate-host weeds (kanghi buti, peeli buti, congress grass, itsit) on bunds and channels to stop whitefly, mealybug, tobacco caterpillar, spotted bollworm spread."),

    Q("Why should I prefer desi cotton in some areas of Punjab?",
      "Prefer desi cotton in areas with high infestation of whitefly and cotton leaf curl disease. Desi cotton is genetically resistant to cotton leaf curl disease and tolerates whitefly better than American cotton. Recommended desi varieties: PBD 88 (2026), LD 1019, LD 949, FDK 124."),

    Q("How can I prevent whitefly outbreaks in my cotton?",
      "Multiple cultural measures: (1) Carry out regular surveillance on alternate hosts (brinjal, cucurbits, tomato, chilli, okra) from February onwards, and on cotton and moong from April onwards. (2) Use 40 yellow sticky traps per acre during initial cotton phase. (3) Eradicate kanghi buti, peeli buti, puth kanda, congress grass, itsit weeds. (4) Avoid excessive nitrogen. (5) Avoid synthetic pyrethroids before September 15 (causes resurgence)."),

    Q("Should I use yellow sticky traps in cotton?",
      "Yes - low-cost yellow sticky traps at 40 per acre during the initial phase of the cotton crop check early infestation of whitefly. Place traps at canopy height, replace regularly. Combined with surveillance on alternate hosts and weed eradication, sticky traps are a key non-chemical whitefly management tool."),

    Q("Can I plant other crops to help suppress mealybug in cotton?",
      "Yes - grow bajra, maize, and jowar as barrier crops around cotton fields. They are the LEAST preferred hosts for mealybug. The pest skips them and is unable to migrate easily. This is part of PAU's integrated mealybug management alongside avoiding bhindi, moong, arhar near cotton (which ARE preferred hosts)."),

    Q("Why should I not throw uprooted infested cotton plants in the field?",
      "Mealybug-infested plants can spread the pest further if dumped in cotton fields or water channels. Burn or bury infested plants away from the field. Also prevent movement of cotton sticks from infested areas to new areas - the pest hitch-hikes on sticks."),

    Q("Should I spray nearby trees and fruit plants for mealybug?",
      "Yes - if trees or fruit plants near cotton fields harbour mealybug populations, spray them with recommended insecticides. The mealybug spreads readily from these reservoirs to cotton. Treating both prevents reinfestation cycles."),

    Q("How do I control tobacco caterpillar egg masses in cotton?",
      "Tobacco caterpillar (Spodoptera litura) lays egg masses on the lower side of mature leaves, covered with brown hairs. Young larvae feed gregariously and skeletonize foliage. Collect egg masses and young larvae along with the leaves they're on, and destroy them physically. This early manual removal prevents the larvae from dispersing and causing wide damage."),

    Q("When and why should I terminate the cotton crop early?",
      "Terminate the cotton crop as early as economically feasible. Give the LAST irrigation by end of September. This reduces bollworm damage and their carryover to next season. Continuing to irrigate and produce late bolls invites pest pressure that hurts current yield AND seeds next season's outbreak."),

    Q("What should I do with cotton field after final picking to manage pink bollworm?",
      "After the final picking, shred the cotton field with a shredder to kill larvae of pink bollworm in unopened bolls. Also destroy all trash collected during ginning, remove all seed from ginneries by end of March, and fumigate uncrushed seed left in mills before end of April with Celphos/Phostoxin/Delicia at one 3 g tablet per cubic metre for 48 hours, or two tablets for 24 hours."),

    Q("Should I keep cotton seed for sowing without acid delinting?",
      "No - apparently healthy seed-cotton (kapas) may harbour pink bollworm larvae. Kapas retained by farmers should be ginned by end of March and the seed fed to cattle. If the seed will be retained for sowing, it must be acid-delinted, fumigated, OR thoroughly sun-dried in a thin layer for 3-4 consecutive days in April. The acid treatment kills pink bollworm larvae and removes fuzz for mechanical sowing."),

    Q("After picking cotton, can I let livestock graze in the field?",
      "Yes - after the last picking, allow sheep, goats, and other farm animals into the cotton fields to feed on plant debris and unopened bolls. They consume sources of pink bollworm larvae and reduce the carryover. This is part of PAU's integrated pink bollworm management strategy."),

    Q("How should I store cotton sticks to manage pink bollworm?",
      "Do NOT stalk cotton sticks under shade or laid flat in the field - this preserves pink bollworm larvae. Beat the sticks on the ground to dislodge larvae from unopened bolls. Stalk the cotton sticks vertically, use or burn them by end of February. Prevent movement of cotton sticks from infested areas to new areas to avoid spreading the pest."),

    # ===== Chunk 07-08: Mating disruption, sucking pests =====
    Q("What is mating disruption based pink bollworm management?",
      "Apply 125 g per acre per application of gossyplure 4 percent paste (CREMIT-PBW or Natamate-PBW) in the form of peanut-size dollops at 400 uniformly distributed spots, starting at square appearance (45-55 days after sowing). Repeat 2 more times at 30-day intervals. Apply paste at the nodal junction of the 5th or 6th main stem leaf from the top. Timely and area-wide application is essential. If it rains within 4-5 hours, repeat the application."),

    Q("How do I apply PB knot for pink bollworm?",
      "Use PB knot (gossyplure 4 percent) in 25-hectare blocks at square formation stage (40-50 days after sowing). Tie wires on upper canopy of cotton plants at 1 metre distance on block borders, and at 5 metre equidistant inside the block. Total 9875 wires per 25 hectare block. PB knot is rain-fast and allows insecticides for other pests during the season. Timely area-wide application is essential."),

    Q("Does Bt cotton control all bollworms?",
      "No - Bt cotton provides effective protection against most bollworms BUT NOT against pink bollworm. Regular weekly monitoring during the reproductive phase is essential. If American bollworm crosses ETL late in the season, use insecticides from PAU's Table 2. Bt cotton also does NOT control sucking pests (whitefly, jassid, mealybug, thrips, aphid) - these require separate management."),

    Q("What sucking pests damage Bt cotton most?",
      "On Bt cotton, the most serious sucking pests are whitefly, jassid, mealybug, thrips, and aphid - all causing maximum damage during July-September. Bt cotton has no resistance to these pests, so all the standard sucking-pest management practices (monitoring, ETL-based spraying, cultural control) apply equally to Bt and non-Bt cotton."),

    Q("How do I identify whitefly damage on cotton?",
      "Whitefly adults and nymphs suck sap from leaves and excrete honey dew, making leaves sticky. Affected leaves and seed cotton turn black due to sooty mould developing on the honey dew. Severe attacks reduce yield substantially and contaminate the lint. Monitor population: spray when 6 adults per leaf appear in the upper canopy before 10 AM, or when honey dew appears on 50 percent of plants."),

    Q("What is the ETL for cotton whitefly?",
      "Spray for whitefly when population reaches 6 adults per leaf in the upper canopy of plants, observed before 10 AM (when whiteflies are less active and easier to count). Alternative trigger: honey dew appears on 50 percent of plants. Below ETL, monitor without spraying - natural enemies handle low populations and unnecessary sprays trigger resurgence."),

    Q("What is the ETL for cotton jassid?",
      "Initiate spray against jassid whenever some of the fully-formed leaves in the upper canopy show curling and yellowing at the margins on 50 percent of plants. Below this threshold, the natural predator population manages jassid without intervention. Seed treatment with Gaucho or Cruiser before sowing prevents early jassid damage."),

    Q("What is the ETL for cotton thrips?",
      "Spray against cotton thrips when population of nymphs and adults reaches 12 per leaf in the upper canopy of plants. Below 12 per leaf, do not spray. NOTE: Do NOT spray any insecticide for thrips on cotton up to 30-day-old crop - if thrips attack is observed, irrigate the field immediately instead. Early thrips sprays disrupt natural enemies and worsen later pest pressure."),

    Q("What is the ETL for cotton aphid?",
      "Spray against cotton aphid on the appearance of honey dew on 50 percent of plants. Aphids appear sporadically and the honey-dew indicator captures economically significant populations. Use the same insecticides recommended for jassid (e.g., Keefun, Osheen, Ulala, Actara) at the listed doses."),

    Q("How do I identify thrips damage on cotton?",
      "Thrips first lacerate leaf tissue then feed on the oozing cell sap. Initial symptom: silver streaks especially around midrib and veins. Later, silvering becomes severe with slight cupping of leaves. Under severe infestation, leaves give a blasted appearance with extreme cupping. ETL: 12 nymphs/adults per leaf in upper canopy. NOTE: never spray thrips up to 30 days; instead irrigate."),

    Q("What insecticides should I use for cotton whitefly?",
      "PAU options for cotton whitefly: 200 g Clasto 20 WG (pyrefluquinazon), or 400 mL Sefina 50 DC (afidopyropen), or 60 g Osheen 20 SG (dinotefuran), or 200 g Polo/Craze/Ruby 50 WP (diafenthiuron), or 500 mL Lano/Daita 10 EC (pyriproxyfen), or 200 mL Oberon 22.9 SC (spiromesifen), or 80 g Ulala 50 WG (flonicamid), or 20 g Dantotsu 50 WG (clothianidin), or 800 mL Fosmite 50 EC (ethion), or 1.0 litre Nimbecidine, or 1200 mL PAU Homemade Neem Extract per acre."),

    Q("What insecticides work for cotton jassid?",
      "For jassid: Seed treatment at sowing with 5 g Gaucho 70 WS or 7 g Cruiser 30 FS per kg seed. For spray on standing crop: 300 mL Keefun 15 EC (tolfenpyrad), or 60 g Osheen 20 SG (dinotefuran), or 300 mL Neon 5 EC (fenpyroximate), or 80 g Ulala 50 WG (flonicamid), or 40 g Actara/Extra Super/Dotara/Thomson 25 WG (thiamethoxam) per acre."),

    Q("What insecticides work for cotton thrips?",
      "For cotton thrips at ETL of 12 per leaf: 240 mL Simodis 10 DC (isocycloseram), or 170 mL Delegate 11.7 SC (spinetoram), or 500 mL Curacron/Celcron 50 EC (profenophos), or 200 g Polo 50 WP (diafenthiuron) per acre. Do NOT spray for thrips on crop younger than 30 days - irrigate instead."),

    Q("What insecticide should I use for cotton mealybug?",
      "For mealybug, spray 150 mL Transform 21.8 SC (sulfoxaflor) per acre as soon as crawlers/adults appear. Mealybug typically starts in patches - spot-treat the affected plants and rows rather than blanket-spraying the whole field. Spray mealybug-infested plants/rows after the last picking too to prevent carryover."),

    Q("What should I do at the start of cotton season for early whitefly?",
      "At the very beginning of the crop season, on first appearance of whitefly, give the first spray of Nimbecidine or Achook (neem-based biopesticide) at 1.0 litre per acre. This is gentle on natural enemies, slows pest build-up, and avoids triggering pyrethroid-style resurgence. Switch to stronger insecticides only when ETL is reached and biopesticides are insufficient."),

    Q("How can I make PAU Homemade Neem Extract for cotton sprays?",
      "Boil 4.0 kg of terminal shoot parts of neem trees (including leaves, green branches, and fruits) in 10 litres of water for 30 minutes. Filter through muslin cloth. Use the filtrate at 1200 mL per acre as a spray for cotton whitefly. This homemade preparation is far cheaper than commercial neem products and gives equivalent early-stage control."),

    Q("Why does PAU say avoid neonicotinoids and pyrethroids in cotton early in the season?",
      "Three reasons: (1) Synthetic pyrethroids (cypermethrin, fenvalerate, deltamethrin), acephate, and acetamiprid trigger whitefly resurgence by killing predators while sparing whiteflies. (2) Early-season use depletes natural enemies that would otherwise control pest build-up. (3) Resistance accelerates with repeated early use. Reserve these for late-season bollworm management AFTER September 15, and rotate active ingredients."),

    # ===== Chunk 09-10: Spray technique, bollworms, monitoring =====
    Q("How should I apply insecticides on cotton for best results?",
      "Five PAU principles: (1) Use fix-type solid cone nozzles. (2) Thorough coverage of plants is essential to check whitefly and mealybug multiplication. (3) Spray before 12 PM or in the evening - avoid mid-day heat. (4) Adopt community approach at village level for area-wide impact. (5) Use only recommended insecticides at recommended dose and time. Avoid tank mixing and readymade insecticidal mixtures."),

    Q("What is the best time of day to spray cotton insecticides?",
      "Spray cotton insecticides BEFORE 12 PM (morning) or in the EVENING. Mid-day spray suffers from heat-induced drift, evaporation, and reduced efficacy. Morning is best for whitefly counts (do them before 10 AM) and for spray when pests are less mobile. Community-level coordination across the village improves area-wide pest control."),

    Q("How do I manage mealybug spot treatment in cotton?",
      "Mealybug is initially restricted to a few plants in a row - so spot treatment with recommended insecticides (150 mL Transform 21.8 SC per acre) on affected plants only is more effective and economical than blanket spraying. After the last picking, spray mealybug-infested plants/rows again to prevent carryover. Combine with cultural measures: barrier crops (bajra, maize, jowar), no bhindi/moong/arhar nearby, kanghi buti and other weed eradication."),

    Q("What are the four major cotton bollworms in Punjab?",
      "Four major bollworms attack cotton in Punjab: (1) Spotted bollworms damage growing points May-June and shed squares, buds, flowers, bolls July-October. (2) American bollworm causes severe shedding of fruiting bodies September-October especially on American cotton. Larvae have one line on upper side and two wavy lines on lateral side. (3) Pink bollworm does maximum damage mid-July to mid-October. (4) Plus tobacco caterpillar (Spodoptera litura) August-October."),

    Q("How do I identify American bollworm larvae on cotton?",
      "American bollworm larvae have variable colour but a distinctive pattern: ONE line on the upper side and TWO wavy lines on the lateral side of the body. The body has sparse hairs. The larvae cause severe shedding of fruiting bodies during September-October, especially on American cotton. ETL-based spray decisions use 5 percent damage in fruiting bodies."),

    Q("How do I identify tobacco caterpillar on cotton?",
      "Tobacco caterpillar (Spodoptera litura) is polyphagous. Small larvae are black; grown-up larvae are dark green with black triangular spots on body. Moths lay egg masses covered with brown hairs on the lower side of mature leaves. First and second instar larvae feed gregariously and skeletonize foliage; later larvae disperse and feed singly. They damage leaves, buds, flowers, and green bolls August-October."),

    Q("How do I monitor cotton bollworms and tobacco caterpillar with traps?",
      "Use sex pheromone traps from flowering stage onwards. Record moth catch on alternate days. For pink bollworm: Sticka/Delta traps with at least 10 microlitres of gossyplure, placed 15 cm above crop canopy, 1 trap per hectare, replace lure every 15 days. For spotted bollworm: Sleeve/Moth catch traps at 15 cm above canopy, 2 traps per hectare. For American bollworm: Sleeve/Moth catch traps with at least 2 mg pheromone, 2 per hectare. Same for tobacco caterpillar."),

    Q("How do I scout for bollworm damage in my cotton field?",
      "Examine fields TWICE A WEEK during the effective boll formation period. Divide the field into 4 quarters; collect 25 freshly-shed fruiting bodies (squares, buds, young bolls) at random in each quarter (100 total). Bollworm-damaged ones have feeding holes or larvae inside. ETL: spray when damage exceeds 5 percent."),

    Q("What is the ETL for cotton bollworms?",
      "ETL for cotton bollworms is 5 percent damage among the freshly-shed fruiting bodies. Examine the field twice a week, collect 100 fresh shed bodies (25 per quarter), check for feeding holes or larvae. Spray immediately if damage exceeds 5 percent, then thereafter as need arises. For very early-sown crop (first half of April), the first pink bollworm spray is at 10-20 percent square production in fields near old cotton sticks or ginneries."),

    Q("Which insecticides should I use for pink and spotted bollworms?",
      "Synthetic pyrethroid options: 300 mL Danitol 10 EC (fenpropathrin), or 100 mL Fastac 10 EC (alphamethrin), or 300 mL Bulldock 0.25 SC (beta-cyfluthrin), or 200 mL Ripcord 10 EC (cypermethrin), or 80 mL Cymbush 25 EC (cypermethrin), or 160 mL Decis 2.8 EC (deltamethrin), or 100 mL Sumicidin 20 EC (fenvalerate) per acre. Use only after September - earlier sprays trigger whitefly resurgence."),

    Q("What is Delegate for bollworm control in cotton?",
      "Delegate 11.7 SC (spinetoram) is a spinosyn-class insecticide. Apply 170 mL per acre for pink, spotted, and younger larvae of American bollworm. Spinosyns have a different mode of action from pyrethroids and organophosphates, supporting resistance management when rotated into the spray plan."),

    Q("How do I use Proclaim (emamectin benzoate) for cotton bollworms?",
      "Proclaim 5 SG contains emamectin benzoate, a macrocyclic lactone (avermectin). Apply 100 g per acre for pink, spotted, and younger larvae of American bollworm. Like Delegate, this gives a different mode of action from older insecticide groups, useful in rotation strategies."),

    Q("What insecticide controls grown-up larvae of American bollworm?",
      "Grown-up American bollworm larvae are tougher to kill - older larvae need: 60 mL Tracer 48 SC (spinosad), or 200 mL Avaunt 15 SC/EC (indoxacarb), or 300 mL Sumipleo 10 EC (pyridalyl), or 60 mL Coragen 18.5 SC (chlorantraniliprole), or 2 litres chlorpyriphos products, or 800 g Orthene 75 SP (acephate) per acre. Chlorpyriphos is preferred for grown-up larvae per PAU's IRM strategy."),

    Q("What insecticides work for tobacco caterpillar in cotton?",
      "Two PAU options: 150 mL Rimon 10 EC (novaluron, an insect growth regulator) per acre, OR 60 mL Coragen 18.5 SC (chlorantraniliprole) per acre. Plus chlorpyriphos, thiodicarb, and quinalphos used for late-season American bollworm also control tobacco caterpillar. Quinalphos at 500 mL of 25 EC in 100 litres water also handles hairy caterpillars in June-July."),

    Q("What is PAU's IRM strategy for cotton sucking pests?",
      "From sowing to first week of July: (1) Sow recommended sucking-pest-tolerant varieties to avoid early sprays. (2) Destroy alternate hosts of whitefly, leaf curl virus, mealybug. (3) Timely sowing, judicious fertilizers/irrigation, proper spacing, clean cultivation prevent early pest build-up and conserve natural enemies. (4) Treat seed with Gaucho or Cruiser for jassid in susceptible cultivars. (5) Do NOT use any insecticide during this period - protect natural enemies."),

    Q("What is PAU's IRM strategy for cotton in mid-season (mid-July to early August)?",
      "Sucking pests and bollworms phase: (1) Avoid synthetic pyrethroids for spotted bollworm control. (2) Avoid neonicotinoid compounds against jassid - they harm natural enemies. (3) Do NOT use organophosphates or carbamates against bollworms during this period. Save those classes for later windows when their use is justified."),

    Q("What is PAU's IRM strategy for cotton in August to October?",
      "Late-season bollworm and tobacco caterpillar window: (1) Mid-August to end August: use profenophos, quinalphos, or flubendiamide alternated with synthetic pyrethroids; spinosad only for severe American bollworm. (2) September to October: use profenophos, quinalphos, thiodicarb, or flubendiamide for younger American bollworm larvae; chlorpyriphos for grown-up larvae. Indoxacarb or spinosad if American bollworm is serious. Use ethion for whitefly - also controls pink and spotted bollworms."),

    # ===== Chunk 13-14: Diseases =====
    Q("What is cotton leaf curl disease?",
      "Cotton leaf curl is caused by a whitefly-transmitted virus. Diseased plants are stunted with twisted internodes. Leaves remain small with cupping and curling. Veins on leaf undersides become thickened with netted appearance. Small leaflets (enations) develop on the undersides of leaves on main and lateral veins. Number of fruiting bodies is reduced. No curative treatment - manage by controlling whitefly vector and using resistant/tolerant varieties (e.g., desi cotton)."),

    Q("How do I manage cotton leaf curl virus disease?",
      "Five integrated measures: (1) Avoid growing American cotton in/around citrus orchards and adjoining bhindi crop. (2) Uproot and destroy diseased plants from time to time. (3) Protect crop against whitefly vector with recommended insecticides. (4) Follow clean cultivation. (5) Destroy collateral hosts like Kanghi buti (Abutilon sp.) and Peeli buti (Sida sp.). Prefer desi cotton in high-pressure areas - it is resistant to leaf curl."),

    Q("What is parawilt in cotton and how do I manage it?",
      "Parawilt is a PHYSIOLOGICAL DISORDER in cotton - no pathogen is involved. It typically occurs after droughts when the crop is heavily irrigated, or after heavy rain. Plants show sudden drooping of leaves which then wilt, but the root system remains intact. Save affected plants by spraying cobalt chloride at 10 mg per litre of water (10 ppm) IMMEDIATELY after symptoms appear. No recovery is possible if permanent wilting has set in."),

    Q("How do I tell parawilt from real wilting in cotton?",
      "In parawilt, the root system remains intact and the leaves droop suddenly after a sequence of drought followed by heavy irrigation or heavy rain. In true fungal wilt (Fusarium, Rhizoctonia, or bacterial wilt), the roots are compromised - they're broken, blackened, or browned in vascular tissues. Parawilt is reversible with cobalt chloride spray (10 ppm) if caught early; fungal wilt is not reversible."),

    Q("What is cotton root rot?",
      "Cotton root rot is caused by Rhizoctonia solani and R. bataticola fungi. Main symptoms: drying and shedding of leaves leading to complete wilting and plant death. The disease spreads in field as round patches. Affected plants can be pulled out very easily. The bark of the roots is broken into shreds. Manage by crop rotation, destroying infected debris, and avoiding waterlogging."),

    Q("What is cotton bacterial blight and how do I prevent it?",
      "Cotton bacterial blight is caused by Xanthomonas axonopodis pv. malvacearum, surviving in seed and plant debris. Lesions on leaves: minute water-soaked angular spots that turn brown then form black angular dead lesions on both sides of the leaf. Bacterium also infects young bolls (small round water-soaked spots depressed in centre). Use disease-free seed - the primary prevention. Acid-delinted seed reduces seed-borne inoculum."),

    Q("What is Myrothecium leaf spot in cotton?",
      "Myrothecium leaf spot is caused by Myrothecium roridum fungus. Appears on leaves, bracts, and bolls as circular to semicircular brown spots with broad violet margins. Later, shield-shaped small fruiting bodies appear in the central necrotic portion. Pathogen is seed-borne and survives on dead leaves. High humidity and intermittent rains favour development. Spray 200 mL Amistar Top 325 SC (azoxystrobin + difenoconazole) per acre in 200 L water on appearance, repeat every 15-20 days if needed."),

    Q("What is Alternaria blight on cotton?",
      "Alternaria gossypina causes leaf blight in cotton. Early spots have a pale-green area with irregular margins. As spots enlarge, irregular concentric zones form. Sometimes severe leaf shedding occurs. Plants with low vigour from drought or potash deficiency are most affected. Disease perpetuates through diseased debris. Spray 200 mL Amistar Top 325 SC per acre in 200 L water on appearance, repeat 15-20 days later if needed."),

    Q("What is Cercospora leaf spot on cotton?",
      "Cercospora leaf spot generally appears toward end of season. Small circular to irregular spots with whitish centre and dark brown margin. In advanced stages, the necrotic central portion may fall out giving a shot-hole appearance. Low temperature (less than 25 degrees C) and high relative humidity favour development. Diseased debris is the main inoculum source. Use disease-free seed. Spray 200 mL Amistar Top 325 SC per acre in 200 L water on appearance, repeat 15-20 days later."),

    Q("What is cotton tirak disorder?",
      "Tirak is a physiological disorder of cotton characterized by yellowing and reddening of leaves followed by bad opening of bolls. Most pronounced in the dry belt adjoining Rajasthan and Haryana. Caused by combinations of: persistent drought, inadequate water, nutrient deficiency on light sandy soils, too-early sowing, lack of plant protection. Spells of high temperature during flowering/fruiting aggravate it. Mitigate with judicious fertilization, timely watering during flowering and fruiting, and recommended plant protection."),

    Q("Should I defoliate my cotton before picking?",
      "Yes - chemical defoliation with a single spray of Ethrel 39 percent (Ethephon 39%) at 5.0 mL per litre of water in the LAST week of October leads to 85-90 percent defoliation after 10 days. Defoliation allows better sunlight penetration, resulting in early and uniform boll opening with increased productivity. The picking is also easier and cleaner."),

    Q("How often should I pick cotton?",
      "Pick cotton clean and dry to get good market price. Picking should be done after every 15-20 days to avoid losses from kapas falling to the ground. Don't keep picked cotton in wet water channels - this impairs quality. Store kapas in a dry godown. Keep different varieties separate."),

    # ===== Chunk 15-16: Cotton stalk uprooter, marketing, desi cotton =====
    Q("What is the cotton stalk uprooter?",
      "Cotton Stalk Uprooter is a two-row tractor-operated implement for removing cotton stalks with their roots after the last picking. Operate at 7-9 km per hour speed and at 12-15 cm depth using a 45 HP tractor. Field capacity: 1.25-1.50 acre per hour. Provides 10-15 percent more cotton sticks by weight than conventional manual stalk chopping. Use or burn cotton sticks by end of February at the latest."),

    Q("How do I dispose of cotton sticks after harvest?",
      "Soon after the last picking, remove cotton sticks along with roots from the field using a tractor-operated cotton stalk uprooter (45 HP tractor, 7-9 km/h, 12-15 cm depth). Bury the remaining plant debris with a furrow turning plough as a sanitary measure against pests and diseases. Use or burn cotton sticks by the end of February. Do NOT stalk sticks under shade or in fields - this preserves pink bollworm larvae."),

    Q("What are PAU's marketing tips for cotton?",
      "Four PAU marketing tips: (1) Pick cotton dry, free from trash, with no dew on it. (2) The first and last pickings are usually low quality and should NOT be mixed with the rest of the produce - mixing reduces overall price. (3) Store cotton in a damp-proof and rat-free room. (4) Store different varieties separately - mixing reduces market grade and price."),

    Q("Why does PAU say not to mix the first and last cotton pickings with the rest?",
      "First and last pickings are usually of low quality - first picking has immature lint and last picking has weather-damaged or pest-affected bolls. Mixing high-grade kapas with low-grade kapas means the entire lot is graded down, selling at a relatively low price. Sell first/last pickings separately at their grade rather than dragging down the bulk of the harvest."),

    Q("How much desi cotton is grown in Punjab?",
      "Desi cotton was grown on 1.7 thousand hectares in Punjab during 2024-25. Total production: 4.8 thousand bales with average lint yield of 4.83 quintals per hectare (1.95 quintal per acre). Desi cotton is much smaller in scale than American cotton (1.59 lakh hectares) but is preferred in areas with high whitefly and leaf curl pressure due to its genetic resistance to leaf curl disease."),

    Q("Tell me about PBD 88 desi cotton variety.",
      "PBD 88 (2026) is a new high-yielding shattering-tolerant desi cotton variety. Average seed cotton yield: 10.9 quintals per acre - higher than other desi varieties. Short-staple coarse fibre with 2.5% span length 20.6 mm and ginning out turn 38.2 percent. Resistant to jassid, whitefly, fusarium wilt, AND bacterial blight - the most disease-resistant desi cotton option."),

    Q("Tell me about LD 1019 desi cotton variety.",
      "LD 1019 (2018) is a shattering-tolerant desi cotton variety needing only 2-3 pickings versus 5+ pickings for other desi varieties. Average seed cotton yield: 8.6 quintals per acre. Ginning outturn 35.7 percent, fibre length 22.6 mm. Green broad leaves, cream flowers. Tolerant to jassid, whitefly, fusarium wilt, and bacterial blight. The reduced picking labour is a major attraction for smaller farms."),

    Q("Tell me about LD 949 desi cotton variety.",
      "LD 949 (2016) is a desi cotton variety with reddish-brown plants, narrow-lobed deep-cut leaves, and pink flowers. Lint percentage 40.1. Fibres are short and coarse, suitable as absorbent cotton. Moderately resistant to fusarium wilt and bacterial blight; tolerant to whitefly and jassid. Average seed cotton yield: 9.9 quintals per acre."),

    Q("Tell me about FDK 124 desi cotton variety.",
      "FDK 124 (2011) is a desi cotton variety with green plant body and narrow-lobed leaves. Synchronous in maturity, takes about 160 days to mature. Short-staple coarse-fibre variety: 2.5 percent span length 21.0 mm, ginning outturn 36.4 percent. Average seed cotton yield: 9.28 quintals per acre. Resistant to jassid and whitefly."),

    Q("How much desi cotton seed do I need per acre?",
      "Use 3.0 kg of desi cotton seed per acre - less than the 3.5 kg used for non-Bt American cotton or the 4.0+1.0 kg used for PAU Bt American cotton varieties. Acid delinting follows the same protocol as American cotton."),

    Q("What plant spacing does desi cotton use within rows?",
      "For desi cotton, sow in rows 67.5 cm apart with a cotton sowing drill. Plant-to-plant spacing within rows should be 45 cm at thinning - between the 30 cm of PAU Bt 2/3 and the 60 cm of non-Bt American cotton. The 45 cm spacing matches desi cotton's mid-sized canopy."),

    Q("When should I do the first bollworm spray on desi cotton?",
      "On desi cotton, the first bollworm spray should be done when 25 percent of plants start producing squares - later than American cotton which uses 5 percent damage in fruiting bodies. Subsequent sprays should be need-based using the same insecticides (Table 2 of PAU recommendations)."),

    Q("My desi cotton plants are growing too tall. What should I do?",
      "Desi cotton on medium to high fertility soils often attains unmanageable height for effective spraying against bollworms. The unsprayed top portion contributes little to yield but greatly helps bollworm build-up. Detop plants taller than 1.5 m using a pruning secateur, sickle, or green mulberry stick as needed. This brings the canopy back into spray reach and reduces pest pressure."),

    Q("What is desi cotton wilt?",
      "Desi cotton wilt is a fungal disease caused by Fusarium oxysporum f.sp. vasinfectum, both soil and seed-borne. Diseased plants: leaves lose turgidity, turn yellow then brown, wilt, and drop. Discoloration starts from leaf margins moving toward midribs. Older leaves first, then younger ones upward. Wilting may be complete or partial (only one side of plant). Diagnostic: browning/blackening of vascular tissues. Manage with 5-6 year rotation with non-host crops, plant tolerant varieties LD 1019 or LD 949, or grow American cotton in highly infested fields (American is wilt-free)."),

    Q("Is desi cotton resistant to leaf curl disease?",
      "YES - desi cotton is resistant to cotton leaf curl disease (the whitefly-transmitted virus that devastates American cotton in some Punjab areas). For this reason PAU specifically recommends preferring desi cotton in areas of high whitefly and leaf curl pressure. For other diseases (wilt, leaf spots, bacterial blight), management is the same as for American cotton."),

    Q("When is desi cotton ready for picking?",
      "Desi cotton is ready for picking in the third week of September. Pick cotton clean and dry. Subsequent pickings should be every 15 days to avoid kapas falling to the ground. Do not keep picked cotton in wet water channels. Store kapas in dry godown. Keep different varieties separate. Removal of cotton sticks and marketing protocols are the same as for American cotton."),

    # ===== Climate-tied general/cross-crop tie-back examples =====
    Q("How does climate change affect cotton viability in Punjab?",
      "Cotton needs 27-32 degrees C day temperature and cool nights during fruiting, plus bright sunny days from mid-September to November for picking. As Punjab warms (CMIP6 projects +1.22 degrees C warming from 2025 to 2050), the September-November window will increasingly see hotter days that compress the picking period and reduce lint quality. Combined with the 32 degrees C upper threshold for proper crop growth, climate warming makes cotton viability marginal in Punjab by 2050. PAU's recommended cotton management may need adjustment as decades pass."),

    Q("Why does early sowing matter for cotton in a warming climate?",
      "PAU recommends sowing cotton by 15 May to escape peak summer heat during establishment AND to push the picking period into cooler late-September weather. As climate warms, this sowing window may need to advance further, and short-duration recommended varieties become more important. The current 4-6 week interval to first irrigation may also shorten as evapotranspiration rises with temperature."),
]

# ===== Write to JSONL =====


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "punjab_agronomy.jsonl"

    # Read existing instructions to check for duplicates
    existing = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["instruction"])
                except json.JSONDecodeError:
                    pass

    # Only append entries whose instruction isn't already in file
    new_entries = [e for e in ENTRIES if e["instruction"] not in existing]
    skipped = len(ENTRIES) - len(new_entries)

    with out.open("a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"Appended {len(new_entries)} cotton entries (skipped {skipped} duplicates)")
    print("Total examples in dataset:")
    subprocess.run(["wc", "-l", str(out)], check=False)


if __name__ == "__main__":
    main()

