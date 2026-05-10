"""Append PAU maize Q&A pairs to the agronomy dataset.

Run: python3 finetune/append_maize.py
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "punjab_agronomy.jsonl"

CATEGORY = "maize"
SOURCE = "pau:maize:kharif_2025"
SUFFIX = "Source: PAU Package of Practices Kharif 2025."


def Q(instruction: str, output: str) -> dict:
    return {
        "instruction": instruction,
        "input": "",
        "output": output + " " + SUFFIX,
        "category": CATEGORY,
        "source": SOURCE,
    }


ENTRIES = [
    # ===== Chunk 01: Production, climate, soil, varieties =====
    Q(
        "How much maize does Punjab grow?",
        "Maize occupied 86.1 thousand hectares in Punjab during 2024-25, with production of 370.5 thousand tonnes and average yield 43.03 quintals per hectare (17.41 quintals per acre). Maize is one of Punjab's major kharif diversification crops, grown for grain, baby corn, sweet corn, popcorn, and silage.",
    ),
    Q(
        "What climate does maize need in Punjab?",
        "Maize needs considerable moisture and warmth from germination to flowering. PAU's productivity windows: germination max/min 36-40/21-27 degrees C, vegetative growth 31-41/24-30 degrees C, flowering to grain filling 31-37/21-27 degrees C. A well-distributed rainfall of 500-700 mm is conducive for good growth. Proper drainage of excess water during heavy rains is essential.",
    ),
    Q(
        "What soil is best for maize in Punjab?",
        "Maize thrives best on well-drained sandy-loam to silty-loam soils. Avoid heavy clays without good drainage and waterlogged soils - flooding particularly at the young stage causes great damage. Drainage is more important than soil richness for maize success in Punjab's kharif rainfall pattern.",
    ),
    Q(
        "What crop rotations work with maize in Punjab?",
        "PAU lists many maize rotations including Maize-Wheat/Barley/Potato/Berseem, Maize-Senji-Sugarcane-Cotton, Maize-Wheat-Moong, Maize-Wheat-Green Manure, Maize-Potato-Toria-Mentha, Maize-Potato-Wheat/Sunflower, Maize-Early Pea-Sunflower, Maize-Wheat-Cowpea fodder, Maize-Raya/Gobhi Sarson, Maize-Potato-Summer Moong, Maize-Potato/Peas-Spring Groundnut, and Maize-Gobhi sarson plus Toria-Mentha relay crop. Most common is Maize-Wheat. Diversification rotations are climate-resilient alternatives to rice-wheat.",
    ),
    Q(
        "Tell me about NK 7328 maize hybrid.",
        "NK 7328 (2026) is a single-cross maize hybrid with tall plants, thick stem, broad erect leaves, and semi-open medium tassel. Ears are long, medium-placed with yellow-orange semi-flint grains. Shows stay-green habit. Matures in 99 days, average grain yield 24.7 quintals per acre. Moderately resistant to maydis leaf blight. Newest PAU long-duration recommendation.",
    ),
    Q(
        "Tell me about PMH 17 maize hybrid.",
        "PMH 17 (2025) is a dual-purpose single-cross maize hybrid suitable for both grain and silage. Tall plants with broad erect leaves, semi-open medium tassel. Ears long, medium-placed with flint yellow-orange capped grains. Matures in 96 days, yields 25.0 quintals per acre. Suitable for ethanol production. Moderately resistant to fall armyworm and maydis leaf blight.",
    ),
    Q(
        "Tell me about DKC 9144 maize hybrid.",
        "DKC 9144 (2024) is a single-cross maize hybrid with tall plants, medium ear height, sturdy stem, broad leaves, and medium open tassel. Ears long with attractive yellow-orange flint grains. Matures in 97 days, yields 24.6 quintals per acre. Moderately resistant to maydis leaf blight, charcoal rot, and fall armyworm - the most disease-resistant among current PAU hybrids.",
    ),
    Q(
        "Tell me about Bioseed 9788 maize hybrid.",
        "Bioseed 9788 (2024) is a single-cross maize hybrid with tall plants, medium ear height, sturdy stem, broad leaves, medium open tassel. Long ears with attractive yellow-orange flint grains. Matures in 96 days, yields 24.3 quintals per acre. Moderately resistant to fall armyworm.",
    ),
    # ===== Chunk 02: More varieties =====
    Q(
        "Tell me about PMH 14 maize hybrid.",
        "PMH 14 (2023) is a single-cross maize hybrid with tall plants, broad erect leaves, semi-open medium tassel, long medium-placed ears with yellow-orange flint capped grains. Matures in 98 days, yields 24.8 quintals per acre. Moderately resistant to fall armyworm. A solid mid-2020s long-duration choice.",
    ),
    Q(
        "Tell me about PMH 13 maize hybrid.",
        "PMH 13 (2021) is a single-cross maize hybrid with tall plants, medium-high ear placement, dark green broad leaves, medium open tassel. Conico-cylindrical long ears with light orange flint grains. Matures in 97 days, yields 24.0 quintals per acre. Moderately resistant to maydis leaf blight, charcoal rot, AND maize stem borer - good multi-stress option.",
    ),
    Q(
        "Tell me about ADV 9293 maize hybrid.",
        "ADV 9293 (2021) is a single-cross maize hybrid with tall plants, medium ear height, sturdy stem, broad leaves. Long ears with attractive orange flint grains. Matures in 97 days, yields 24.5 quintals per acre. Moderately resistant to maydis leaf blight, charcoal rot, and maize stem borer - private hybrid with comprehensive resistance.",
    ),
    Q(
        "Tell me about JC 12 maize composite variety.",
        "JC 12 (2020) is a composite maize variety with medium-tall plants, medium ear placement, medium-thick lodging-resistant stem, heavy open tassel. Ears medium-long with good girth and semi-flint yellow-orange grains. Matures in about 99 days, yields 18.2 quintals per acre - lower than hybrids but with the advantage that farm-saved seed performs well. PAU specifically recommends JC 12 for the kandi (rainfed sub-mountain) areas of Punjab.",
    ),
    Q(
        "Which maize variety is best for kandi areas of Punjab?",
        "JC 12 (2020) is specifically recommended by PAU for kandi areas of Punjab. It is a composite variety, so farm-saved seed retains 80-85 percent of yield potential year over year - a major cost advantage in low-investment kandi farming. Plant height is medium, lodging-resistant, matures in 99 days, yields 18.2 quintals per acre. JC 4 is also recommended for both irrigated and kandi areas.",
    ),
    Q(
        "Tell me about PMH 11 maize hybrid.",
        "PMH 11 (2019) is a single-cross maize hybrid with tall plants and well-developed root system. Sturdy green stem, light green broad leaves, open heavy tassel. Ears long with dark orange flint grains. Matures in 95 days, yields 22.0 quintals per acre.",
    ),
    Q(
        "Tell me about PMH 1 maize hybrid.",
        "PMH 1 (2005) is one of PAU's reliable older single-cross maize hybrids. Tall plants with well-developed root system, zig-zag sturdy purple-coloured stem, medium broad leaves, open medium-sized tassel. Medium-long ears with yellow-orange flint grains. Stays green at maturity. Matures in 95 days, yields 21.0 quintals per acre. Long-standing PAU benchmark for evaluating newer hybrids.",
    ),
    Q(
        "Tell me about JC 4 maize composite variety.",
        "JC 4 (2021) is a composite maize variety with medium-tall plants, medium-placed long ears, deep-orange bold grains. PAU rates it 'very good' for chappati quality (taste, texture, appearance, flavour). Matures in about 90 days, yields 13.0 quintals per acre. Recommended for both irrigated AND kandi areas. Also recommended for organic farming - one of the few PAU varieties with explicit organic-farming endorsement.",
    ),
    Q(
        "Which maize variety is best for chappati making?",
        "JC 4 (2021) is rated 'very good' by PAU for chappati quality parameters - taste, texture, appearance, and flavour. It is a composite variety with medium-tall plants, deep-orange bold grains, matures in 90 days, yields 13.0 quintals per acre. Lower-yielding than hybrids but the chappati quality and organic-farming suitability make it valuable for household consumption.",
    ),
    Q(
        "Tell me about PMH 2 short-duration maize hybrid.",
        "PMH 2 (2005) is a short-duration single-cross maize hybrid - the shortest in PAU's recommendations. Medium plant height, medium ear placement, medium-sized dark green leaves, semi-open medium tassel. Medium-long ears with yellow-orange flint grains and yellow caps. Resists lodging and is tolerant to bacterial stalk rot. Matures in about 83 days, yields 18.0 quintals per acre. Also drought-tolerant - PAU's preferred hybrid for rainfed conditions.",
    ),
    # ===== Chunk 03: Special purpose, agronomy, sowing =====
    Q(
        "What is Punjab Baby Corn 1 and how do I grow it?",
        "Punjab Baby Corn 1 (2022) is a single-cross male-sterile maize hybrid most suitable for baby corn. Being male-sterile means NO detasseling is needed - a major labour saving. Picking starts around 52 days after sowing, and the hybrid gives about 3 pickings per plant. Average yield: 8.4 quintals per acre of dehusked baby corn ears, plus 128 quintals per acre fodder yield after picking is complete. Combined revenue from baby corn plus fodder makes it more profitable than grain maize on small areas.",
    ),
    Q(
        "Tell me about Punjab Sweet Corn 1.",
        "Punjab Sweet Corn 1 (2008) is a composite sweet corn variety with tall plants, medium-thick stem, medium ear placement. Broad leaves, open tassel with creamish anthers, medium-long white ears, creamish silk colour. Well-developed husk, orange grains at maturity. Highly suitable for commercial sweet corn use - immature green ears have high sugar content. Matures in 95-100 days. Average green ear yield: 50.0 quintals per acre; mature grain yield 13.0 quintals per acre.",
    ),
    Q(
        "What is Pearl Popcorn?",
        "Pearl Popcorn (1995) is a composite popcorn variety. Long thin ears with small round grains. Commercial value is high because of good popping quality. Matures in about 88 days, average yield 12.0 quintals per acre. Use 7 kg seed per acre - less than the 10 kg used for grain maize.",
    ),
    Q(
        "How do I prepare the field for maize sowing?",
        "Give 4-5 ploughings and plankings to make the seedbed free from clods and weeds. Use mould-board plough, disc-harrow, or cultivator for the first cultivation. Level the field for proper irrigation and drainage. Alternatively, maize can be sown without preparatory tillage using a zero-till drill - especially after wheat in maize-wheat rotation.",
    ),
    Q(
        "When should I sow maize in Punjab?",
        "Last week of May to end of June for irrigated maize. In fields prone to water-stagnation damage, sow in end-May to early-June so the crop establishes firmly before rains. Sowing at the right time gives higher yield AND vacates the field on time for toria/potato sowing. For rainfed maize, sow June 20 to July 7 (as early as possible after the rains).",
    ),
    Q(
        "How much maize seed do I need per acre?",
        "Use 10 kg seed per acre for most maize varieties. Exceptions: 7 kg per acre for Pearl Popcorn (smaller grains), 20 kg per acre for baby corn (Punjab Baby Corn 1) due to higher row density. The pneumatic planter uses 9-10 kg per acre at 55-56 mm depth.",
    ),
    Q(
        "How do I inoculate maize seed with biofertilizer?",
        "Mix half kg packet of recommended consortium biofertilizer with 1 litre of water. Then thoroughly mix it with maize seed on a clean pucca floor. Let it dry in shade. Sow the seed immediately. Consortium biofertilizer increases grain yield AND improves soil health. Available at PAU Seed Shop Gate No. 1, Krishi Vigyan Kendras, and Farm Advisory Service Centres in different districts.",
    ),
    # ===== Chunk 04: Sowing methods, spacing, intercropping =====
    Q(
        "What spacing should I use for maize in Punjab?",
        "Sow seed 3-5 cm deep in lines with a maize planter or seed-cum-fertilizer drill. Row-to-row spacing 60 cm, plant-to-plant spacing 20 cm. This gives target plant population of 33,333 plants per acre. Bed sowing uses 67.5 cm row spacing with 18 cm plant spacing on top of bed, OR 60 cm spaced ridges with 20 cm plant spacing 6-7 cm above the base on the side.",
    ),
    Q(
        "What is trench sowing for maize?",
        "Trench sowing uses tractor-drawn ridger to make trenches from end-May to mid-June. Sow seed at the base of the trench. Maize raised in trenches resists lodging and gives higher grain yield than flat sowing. The trenches also facilitate easy and economical irrigation during dry hot conditions. A seed drill attachment mounted on the ridger can do trench-making and sowing in one pass.",
    ),
    Q(
        "Should I sow maize on beds or flat?",
        "PAU recommends bed/ridge sowing to avoid the adverse effect of excess rainfall, particularly at seedling emergence (a vulnerable stage). Sow seed 3-5 cm deep on top centre of beds with 67.5 cm row spacing and 18 cm plant spacing. OR sow 6-7 cm above the base on the side of 60 cm spaced ridges, plant spacing 20 cm. Wheat bed planter can be used for bed preparation.",
    ),
    Q(
        "Can I do zero-tillage sowing of maize?",
        "Yes - maize can be grown without preparatory tillage using a zero-till drill, especially after conventional or zero-till wheat. If the field is weedy, control weeds first by spraying half litre of Gramoxone 24 SL (paraquat) per acre in 200 litres of water before sowing. Zero tillage saves fuel, time, and preserves soil structure - particularly valuable in maize-wheat rotation.",
    ),
    Q(
        "Can I intercrop maize with other crops?",
        "Yes - PAU recommends intercropping maize at 60 cm row spacing with one row of: cowpea or maize as fodder, soybean for grains, or groundnut for pods. Apply recommended fertilizers to maize and to intercrops on area basis. Harvest cowpea/maize fodder at 45-55 days after sowing. Intercropping gives higher productivity and monetary returns than sole maize.",
    ),
    Q(
        "How do I thin my maize crop?",
        "If sowing was not done with a planter, thin out plants at the time of first hoeing to maintain plant-to-plant distance of 20 cm. Target plant population is 33,333 plants per acre (60 cm row spacing × 20 cm plant spacing). Over-thinning reduces yield; under-thinning causes intra-row competition.",
    ),
    Q(
        "How do I control weeds in maize culturally?",
        "Two cultural options: (1) Give two hoeings 15-30 days after sowing using khurpa, kasaula, wheel-hoe, triphali, or tractor-drawn cultivator. (2) Spread 30 quintals per acre of paddy straw mulch at sowing - the mulch effectively controls annual weeds AND turns paddy straw into a productive input. (3) Grow 1-2 rows of cowpea (CL 367, 8 kg seed/acre) between maize rows; harvest as fodder at 35-45 days. After cowpea harvest, no further weed control is needed.",
    ),
    # ===== Chunk 05: Chemical weed control, fertilizers =====
    Q(
        "What is atrazine and how do I apply it to maize?",
        "Atrazine (sold as Atrataf, Atragold, Masstaf, Atari, Traxx 50 WP) is a pre-emergence herbicide effective against annual grasses and broadleaf weeds especially itsit. Spray 800 g per acre on medium to heavy textured soils, OR 500 g per acre on light soils, within 10 days of sowing using 200 litres of water. Alternatively, spray 250 g per acre on a 20 cm wide band over crop rows followed by hoeing at 15-30 days.",
    ),
    Q(
        "What is Laudis herbicide for maize and when do I apply it?",
        "Laudis 420 SC contains tembotrione - a post-emergence herbicide for maize. Spray 105 mL per acre in 150 litres of water at 20 days after sowing. Provides effective control of mixed weed flora, useful when atrazine wasn't applied or weeds escaped pre-emergence control.",
    ),
    Q(
        "How do I control dila/motha weed in maize?",
        "Apply 400 mL per acre of 2,4-D amine salt 58 SL as post-emergence at 20-25 days after sowing in 150 litres of water. This specifically controls dila and motha (sedge weeds) which are not well controlled by atrazine. Take care not to drift onto cotton in adjacent fields - cotton is highly sensitive to 2,4-D.",
    ),
    Q(
        "Should I do green manuring before maize?",
        "Yes - PAU recommends green manuring with dhaincha, sunhemp, or cowpea before maize. Sow during the second fortnight of April using 12 kg cowpea OR 20 kg dhaincha OR 20 kg sunhemp seed per acre. The 50-day-old green manure should be buried and allowed to decompose for about 10 days before maize sowing. Apply full nitrogen dose (50 kg N per acre) to maize after green manuring for high yield AND improved soil health.",
    ),
    Q(
        "How much nitrogen does maize hybrid need on medium fertility soil?",
        "For long-duration hybrids (PMH 1, PMH 11, PMH 13, PMH 14, PMH 17, NK 7328, DKC 9144, Bioseed 9788, ADV 9293, JC 12, Punjab Sweet Corn 1): apply 50 kg N, 24 kg P2O5, and 12 kg K2O per acre. Sources: 110 kg urea, plus 55 kg DAP or 150 kg single superphosphate or 125 kg nitrophosphate, plus 20 kg muriate of potash. Apply potash only if soil tests low.",
    ),
    Q(
        "How much fertilizer does PMH 2 short-duration maize need?",
        "For short-duration maize varieties (PMH 2, JC 4, Pearl Popcorn): apply 35 kg N, 12 kg P2O5, and 8 kg K2O per acre. Sources: 75 kg urea, plus 27 kg DAP or 75 kg SSP or 62 kg nitrophosphate, plus 15 kg muriate of potash. Lower than long-duration hybrid rates because of shorter growing cycle and lower yield potential.",
    ),
    # ===== Chunk 06: Fertilizer rules, LCC =====
    Q(
        "Should I apply phosphorus to maize after wheat?",
        "No - if maize follows wheat that received the recommended dose of phosphorus, OMIT phosphorus on maize. Wheat is more phosphorus-responsive than maize, and residual P meets maize's needs. When using 27 kg DAP, reduce urea by 10 kg; when using 55 kg DAP, reduce urea by 20 kg. When using 125 kg nitrophosphate, reduce urea by 50 kg; when using 62 kg nitrophosphate, reduce urea by 25 kg.",
    ),
    Q(
        "How should I split nitrogen on maize?",
        "Drill ONE-THIRD of nitrogen plus the entire phosphorus and potassium at sowing. If using nitrophosphate, omit urea at sowing. Top dress one-third of nitrogen at the knee-high stage (~30 days). Apply the remaining one-third at the pre-tasseling stage (~50-60 days). Three splits match maize's growth-stage nitrogen demand and reduce leaching losses in the rainy kharif season.",
    ),
    Q(
        "How do I use PAU-LCC for need-based nitrogen on maize?",
        "Apply basal dose of 25 kg urea per acre at sowing. Start matching colour of the first fully-exposed leaf from top with PAU-LCC at 10-day intervals starting 21 days after sowing. If 6 or more out of 10 leaves are LIGHTER than LCC shade 5, apply 25 kg urea per acre. If colour equals or is darker than shade 5, apply NO urea. Discontinue LCC after silking initiation - no more urea after that. Match leaves under shade of body, ensure no insect/disease/water-stress.",
    ),
    Q(
        "How does maize fertilizer change with fish pond sediments?",
        "If maize is sown after applying 6 tonnes per acre of fish pond sediments to the field, apply 25 percent LESS fertilizer than the standard recommendation. Fish pond sediments are nutrient-rich and supplement chemical fertilizer. Same 25 percent reduction applies to nitrogen, phosphorus, and potassium.",
    ),
    Q(
        "My maize has white stripes on leaves with reddish veins. What's wrong?",
        "These are classic zinc deficiency symptoms in maize, appearing within 2 weeks of seedling emergence. A broad band of white or light-yellow tissue with reddish veins develops on each side of the midrib, beginning at the base of the second or third leaf from the top. The white patch extends in stripes parallel to the midrib. Plants stay stunted with short internodes. Apply 10 kg zinc sulphate heptahydrate (21 percent) OR 6.5 kg zinc sulphate monohydrate (33 percent) per acre mixed with equal dry soil along rows; hoe in and irrigate.",
    ),
    # ===== Chunk 07: Zinc deficiency, irrigation =====
    Q(
        "How do I correct zinc deficiency in maize during the season?",
        "If zinc deficiency appears in standing maize, apply 10 kg zinc sulphate heptahydrate (21 percent) OR 6.5 kg zinc sulphate monohydrate (33 percent) per acre, mixed with equal quantity of dry soil along rows. Hoe into the soil and irrigate. If symptoms appear too late for interculture, foliar-spray a zinc sulphate-lime mixture: 1.2 kg zinc sulphate heptahydrate plus 0.6 kg unslaked lime, OR 0.75 kg zinc sulphate monohydrate plus 0.38 kg unslaked lime, in 200 litres of water per acre.",
    ),
    Q(
        "How many irrigations does maize need?",
        "Generally 4-6 irrigations are required depending on rainfall. Adequate water is essential throughout the season but PARTICULARLY during pre-tasseling, silking, and grain-filling stages. Water stress at these critical stages causes severe yield loss. Drip irrigation and fertigation are also viable for maize, especially in maize-wheat-summer moong rotations on permanent beds.",
    ),
    Q(
        "What are the most critical water stages for maize?",
        "Pre-tasseling, silking, and grain-filling stages are the most water-sensitive periods for maize. Stress at these stages causes severe shedding of pollen, poor kernel set, and shrivelled grain - all directly hitting yield. Plan irrigations to maintain adequate soil moisture through these windows. Combined with the warming Punjab climate, ensuring water through these stages is increasingly important.",
    ),
    Q(
        "My maize field is flooding from heavy rain. What should I do?",
        "Drain excess water immediately by making a drain of adequate capacity at the lower end of the field. Maize tolerates heavy rains IF not subjected to prolonged excessive wetness or flooding. Flooding particularly at the young stage causes great damage. After flooding subsides, if damage is moderate, spray 6 kg urea per acre as 3 percent solution in two sprays at weekly interval. If moderate to severe, broadcast 12-24 kg additional nitrogen (25-50 kg urea) per acre.",
    ),
    Q(
        "When should I harvest my maize crop?",
        "Maize is ready for harvest when stalks and leaves are still somewhat green BUT the husk cover has dried and turned brown. In fields where wheat is to be sown, harvest the stalks along with cobs and stack them. For better shelling results with maize dehusker-cum-thresher, shell when moisture is between 15-20 percent. Conventional sheller can be used after removal of ears. After shelling, dry grains to about 15 percent moisture for marketing.",
    ),
    Q(
        "What is the optimum moisture content for maize storage?",
        "After shelling, dry maize grains to about 15 percent moisture content for safe marketing and storage. For seed purposes, dry to lower moisture for longer storage. PAU's portable maize dryer (3-tonne capacity) can dry maize from 25 percent to 15 percent moisture in 8-10 hours, maintaining 60-75 degrees C air temperature with 45 degrees C grain temperature for seed (60 degrees C for commercial). Combine harvesting is also possible, but dry ears 3-4 days first.",
    ),
    # ===== Chunk 08: Maize dryer, baby corn =====
    Q(
        "What is baby corn and how do I grow it?",
        "Baby corn is the young ear of female maize inflorescence harvested before fertilization, when silks have just emerged. Used as salad, vegetable, in pickle, pakora, soup. Has export potential. The crop is completed in 60-65 days, much shorter than grain maize. Sowing window: April to first week of August - allows 2+ crops per year. Sow at 30 cm × 20 cm spacing using 20 kg seed per acre. Apply 24 kg N (52 kg urea) per acre in 2 splits at sowing and knee-high. Use Punjab Baby Corn 1 (8.4 q/acre) or Parkash (7.0 q/acre) hybrids.",
    ),
    Q(
        "How fast is the baby corn crop?",
        "Baby corn is completed in 60-65 days from sowing - much shorter than grain maize at 95-99 days. This means you can take 2 or more baby corn crops per year from the same piece of land. Combined with sowing flexibility (April to first week of August), staggered sowing maintains supply as per market demand. PAU specifically recommends staggered sowing to match supply with demand.",
    ),
    Q(
        "How much seed and fertilizer does baby corn need?",
        "Sow 20 kg baby corn seed per acre at 30 cm row × 20 cm plant spacing - higher density than grain maize. Apply 24 kg N (52 kg urea) per acre in two equal splits: at sowing and at knee-high stage. Use PAU-LCC for need-based urea: basal 18 kg urea per acre at sowing; if 6+ of 10 leaves lighter than LCC shade 5, apply 18 kg urea per acre at 10-day intervals starting 21 days after sowing (28 days for winter sowing). Discontinue LCC after silking.",
    ),
    Q(
        "How do I pick baby corn properly?",
        "Pick baby corn ears at SILK EMERGENCE stage. Ears picked later become pithy, woody, and poor quality. Take only 3 picks from each plant - ears appearing later are not good quality. For Parkash hybrid, remove tassels as soon as they appear to prevent pollination. For Punjab Baby Corn 1, no detasseling is needed (it's male sterile). Take ears with single husk layer to market after dehusking.",
    ),
    Q(
        "Tell me about Parkash maize hybrid.",
        "Parkash (1997) is an early-maturing single-cross maize hybrid with medium-tall plants, medium ear placement, dark green medium-sized semi-erect leaves, medium open tassel. Short anthesis-silking interval gives drought tolerance. Uniform long ears with slightly blank tip, attractive orange flint grains, thin white cob. Stay-green characteristic. Matures in 82 days, yields 15 quintals per acre. Recommended for rainfed conditions and also viable for baby corn at 7.0 quintals per acre.",
    ),
    Q(
        "Which maize varieties are best for rainfed Punjab conditions?",
        "PAU recommends two varieties for rainfed (kandi/dryland) maize. PMH 2 (2005) is short-duration drought-tolerant hybrid, matures in 82 days, yields 16.5 quintals per acre under rainfed with well-distributed rains. Resists lodging, tolerant to bacterial stalk rot. Parkash (1997) is early-maturing with short anthesis-silking interval giving drought tolerance, matures 82 days, yields 15 q/acre. Both should be sown June 20 to July 7 - as early as possible after rains.",
    ),
    # ===== Chunk 09-10: Rainfed maize, moisture conservation =====
    Q(
        "How can I conserve moisture for rainfed maize?",
        "Five PAU practices: (1) Repair field bunds and do minor levelling before onset of rains. (2) Plough against the slope after pre-monsoon showers to enhance rainwater absorption/infiltration. (3) Sowing and other operations on contour, across the slope. (4) Spread locally available mulching material in standing crop in the last week of August. (5) Use early-maturing drought-tolerant hybrids (PMH 2, Parkash) sown as early as possible in the June 20 - July 7 window.",
    ),
    Q(
        "How much fertilizer does rainfed maize need?",
        "For sandy-loam to clay-loam soils with adequate moisture stored: 32 kg N, 16 kg P2O5, 8 kg K2O per acre (70 kg urea, 35 kg DAP or 100 kg SSP, 15 kg MOP). For loamy-sand to sandy soils with low moisture stored: 16 kg N, 8 kg P2O5, 4 kg K2O per acre (35 kg urea, 18 kg DAP or 50 kg SSP, 8 kg MOP). Drill half N plus all P and K at sowing; top dress remaining half N one month after sowing. Apply potash only if soil tests low.",
    ),
    Q(
        "What is maize stem borer and how do I manage it?",
        "Maize stem borer is a serious pest from June to September. Larvae first scrape leaves then bore into the stem through the whorl or leaf sheath. Central whorl leaves get perforated. In young plants, the growing point is killed causing a dead-heart. Integrated control: (1) Plough fields after harvest, destroy stubbles, stalks, cobs by end of February. (2) Remove and destroy plants showing severe injury during hoeing. (3) Use trichocards (Trichogramma chilonis parasitizing Corcyra eggs) - 40,000 eggs per acre, twice. (4) Spray 30 mL Coragen 18.5 SC (chlorantraniliprole) in 60 litres water per acre with knapsack sprayer 2-3 weeks after sowing on noticing borer injury.",
    ),
    # ===== Chunk 11: Fall armyworm, other pests =====
    Q(
        "How do I identify fall armyworm in maize?",
        "Fall armyworm larvae have three diagnostic features: (1) White-coloured inverted Y-shaped mark on the head. (2) Four spots arranged in a square pattern at the tail end. (3) Predominant feeding pattern - young larvae scrape leaf surface making papery windows; bigger larvae feed voraciously on central whorl leaves causing round to oblong holes plus large amount of faecal matter. Confirm identification before spraying.",
    ),
    Q(
        "How do I control fall armyworm in maize?",
        "Five-step PAU plan: (1) Sow at recommended time only. (2) Avoid staggered sowing in adjacent fields - this minimizes pest spread. (3) Regularly monitor and destroy egg masses (covered with hairs, easily visible). (4) For crop up to 20 days old, spray Coragen 18.5 SC at 0.4 mL per litre, OR Delegate 11.7 SC at 0.5 mL per litre, OR Missile 5 SG at 0.4 g per litre, in 120 litres water per acre, directing nozzle towards the whorl. (5) For older crop, increase water to 200 litres with proportional dose increase.",
    ),
    Q(
        "What if fall armyworm is in patches or my maize is over 40 days old?",
        "If infestation is in patches OR crop is more than 40 days old (spraying becomes difficult through tall canopy), apply soil-insecticide/biopesticide MIXTURE in the whorls of infested plants. Prepare mixture: 5 mL Coragen 18.5 SC, OR 5 mL Delegate 11.7 SC, OR 5 g Missile 5 SG, OR 25 g Delfin WG (Bt kurstaki), OR 25 mL Dipel 8L (Bt kurstaki) - in 10 mL water mixed well into 1 kg soil. Apply about half a gram per whorl. Use gloves.",
    ),
    Q(
        "What other insect pests attack maize?",
        "Beyond the major two (stem borer, fall armyworm), maize is attacked by: jassid, thrips, pyrilla, grey weevil, leaf-feeding insects (kharif crop), armyworm and silk cutter (whorl feeders - control with same insecticides as maize borer), hairy caterpillars (gregarious, can be epidemic - destroy with light traps and physical removal), and mites (June or September-October, dusty webs visible on pale leaves).",
    ),
    Q(
        "How do I manage hairy caterpillars in maize?",
        "Three measures: (1) Use light traps for destruction of moths. (2) Young larvae are gregarious - destroy them by plucking infested leaves or pulling out infested plants and burying them. (3) Destroy grown-up caterpillars by crushing under feet or by picking and putting them into kerosenized water. Hairy caterpillars can become an epidemic, so early detection and physical control matter.",
    ),
    # ===== Chunk 12: Diseases =====
    Q(
        "What is maize seed rot and how do I prevent it?",
        "Seed rot and seedling blight (caused by several fungi) in maize causes poor germination, unthrifty seedlings, and seedling mortality. The primary prevention is using DISEASE-FREE SEED. Combined with good seed treatment (per recommendations) and proper sowing depth (3-5 cm), seed rot risk is minimized.",
    ),
    Q(
        "What is banded leaf and sheath blight in maize?",
        "Banded leaf and sheath blight is caused by Rhizoctonia solani fungus. Symptoms: water-soaked, straw-coloured necrotic lesions alternating with dark brown bands on basal leaf sheaths. Lesions enlarge and coalesce. Sclerotia (small dark fungal bodies) develop on diseased sheaths, husk, and cobs. In severe cases, developing ears are completely damaged and dry up prematurely with cracking husk. Spray 100 mL Amistar Top 325 SC (azoxystrobin + difenoconazole) per acre in 200 litres of water at disease appearance; repeat at 15 days if needed.",
    ),
    Q(
        "What is maydis leaf blight in maize?",
        "Maydis leaf blight is caused by Drechslera maydis fungus. Spindle-shaped necrotic to brown lesions appear on leaves; lesions merge to form large irregular patches. Symptoms also appear on leaf sheaths, cob husks, and ears. Late sowing, high humidity (above 80 percent), and 25 plus or minus 2 degrees C favours the disease. Manage by destroying infected crop residue, growing improved varieties (NK 7328, PMH 17, DKC 9144 are moderately resistant), and spray as for brown stripe downy mildew.",
    ),
    Q(
        "What is bacterial stalk rot in maize?",
        "Bacterial stalk rot is caused by Dickeya zeae bacterium. Symptoms: water soaking and rotting of basal stem (especially leaf sheaths) followed by rapid rotting of basal internodes. Rind loses natural green colour and looks 'boiled in water'. Rotten stalks emit characteristic fermenting odour and may break at second or third basal internode. Infected plants wilt. Manage by destroying diseased plant debris and ensuring fields are well-drained - excessive rain and poor drainage favour the disease.",
    ),
    Q(
        "What is brown stripe downy mildew in maize?",
        "Brown stripe downy mildew is caused by Sclerophthora rayssiae var zeae. Symptoms: long, narrow, brownish, interveinal stripes on leaves. On close examination, whitish downy fungal growth visible on underside of stripes. Manage: (1) Destroy collateral host Takri grass (Digitaria sanguinalis) from the maize field. (2) Keep fields well-drained. (3) Spray 200 g Indofil M-45 (mancozeb) per acre in 100 litres of water about 2 weeks after sowing, then 2 more sprays at 10-day intervals. (4) Grow recommended varieties.",
    ),
    Q(
        "What is post-flowering stalk rot in maize?",
        "Post-flowering stalk rots are caused by Fusarium spp., Macrophomina spp., and Cephalosporium spp. fungi. Plants wilt after flowering. Rind and basal internodes become discoloured. Splitting the stalk shows discolouration of pith progressing upward. Manage by growing improved varieties such as PMH 13, PMH 11, and PMH 1 - all moderately resistant. Avoiding late-season water stress also reduces severity.",
    ),
    # ===== Chunk 13-14: Hybrid seed production, climate links =====
    Q(
        "Can I save my own hybrid maize seed?",
        "No - hybrid maize seed cannot be saved year-over-year. PAU specifies fresh hybrid seed must be obtained every year from PAU or Punjab State Seed Corporation. If you use grain produce of a hybrid as seed, you will get 15-20 percent LESS yield. Composite varieties (JC 4, JC 12, Pearl Popcorn, Punjab Sweet Corn 1, J 1006, J 1007, J 1008) CAN be used as seed for 3-4 years without major reduction - that's their advantage over hybrids.",
    ),
    Q(
        "How do composite maize varieties differ from hybrids?",
        "Composite varieties (JC 4, JC 12, Pearl Popcorn, Punjab Sweet Corn 1) yield 13-18 quintals per acre - lower than hybrids (22-25 q/acre). BUT their farm-saved grain can be used as seed for 3-4 years without major reduction in yield. Hybrid seed must be purchased fresh every year. For low-investment kandi farming, composites are economically sensible. JC 12 is specifically recommended for kandi areas; JC 4 for both irrigated and kandi (with organic farming approval).",
    ),
    Q(
        "How do I maintain seed purity of composite maize varieties?",
        "Three precautions to maintain composite variety purity and production potential: (1) Avoid admixture with other varieties during sowing, harvest, and storage. (2) Avoid natural cross-pollination - isolate composite plot 200 metres from other maize, OR grow 1 acre composite and select ears from central portion leaving 9-metre strip. (3) Take about 5,000 maize ears and mix grains for next year's seed - never bulk less than 3,000 ears even for small needs.",
    ),
    Q(
        "Why is maize a good climate-adaptation crop for Punjab?",
        "Maize is one of PAU's recommended diversification crops away from rice-wheat, with several climate advantages: (1) Lower water requirement than rice (about 500-700 mm vs 1500+ mm for puddled rice). (2) Shorter cycle (95-99 days for grain hybrids, 60-65 days for baby corn) allows better fit with shifting monsoon timing. (3) Multiple-use options (grain, baby corn, sweet corn, popcorn, silage) give market flexibility. (4) Drought-tolerant hybrids (PMH 2, Parkash) viable in rainfed and kandi areas where rice fails. (5) Compatible with zero-tillage after wheat - reducing fuel and labour.",
    ),
    Q(
        "Should I switch from rice to maize on water-stressed fields?",
        "Maize is one of PAU's recommended diversification options for water-stressed Punjab fields. Maize uses only 500-700 mm of water versus 1500+ mm for puddled rice. PMH 2 and Parkash are drought-tolerant hybrids viable even in rainfed conditions. As Bathinda's groundwater declines (you've seen rainfall drop 28 percent over 25 years) and CMIP6 projects further +1.22 degrees C warming by 2050, switching some rice acreage to maize is a real long-term option. Maize-Wheat-Summer Moong rotation is a productive water-efficient alternative to Rice-Wheat.",
    ),
    Q(
        "What's the most water-efficient kharif crop option for Punjab?",
        "Among PAU's recommended kharif crops, maize is one of the most water-efficient at 500-700 mm rainfall requirement (versus 1500+ mm for puddled rice). Cotton needs 4-6 irrigations and is similarly efficient. Pulses (moong, arhar) are even less demanding. The most water-saving rotation alternatives to Rice-Wheat are: Maize-Wheat-Summer Moong, Maize-Wheat-Green Manure, and Cotton-Wheat. Combined with short-duration varieties, drip irrigation, and laser-levelled fields, water savings of 40 percent or more are achievable.",
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

    print(f"Appended {len(new_entries)} maize entries (skipped {skipped} duplicates)")
    print("Dataset now contains:")
    subprocess.run(["wc", "-l", str(OUT)], check=False)


if __name__ == "__main__":
    main()
