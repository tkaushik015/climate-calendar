# ClimateCalendar

**Climate-adapted planting companion for smallholder farmers — Gemma 4 Good Hackathon 2026**

ClimateCalendar gives smallholder farmers a per-GPS, per-soil, per-ENSO planting recommendation grounded in 30+ years of weather history, CMIP6 climate projections to 2050, and the farmer's actual soil-test report. Built for Ramesh Singh — a wheat farmer in Bathinda, Punjab — and the millions like him whose families have planted on the same date for generations, while the climate beneath their feet has quietly shifted.

## What it does

- **Listens** in the farmer's own voice (Whisper, 99 languages including Hindi/Punjabi)
- **Reads** the farmer's soil-test report from a phone photo (EasyOCR + structured parser)
- **Reasons** across real climate data using a Gemma 4 function-calling agent
- **Recommends** a 5–7 day planting window grounded in cited data — never hallucinated

## Architecture

The agent loop: farmer query → Gemma 4 chooses which tools to call → Python executes → Gemma synthesizes a grounded answer citing specific numbers.

### Tools (`src/tools/`)

| Tool | Source | What it returns |
|------|--------|-----------------|
| `climate_trend` | Open-Meteo ERA5 (1940–present) | Yearly temperature & rainfall for any GPS |
| `climate_projection` | Open-Meteo CMIP6 (MRI-AGCM3-2-S) | Projected climate to 2050 |
| `enso` | NOAA Oceanic Niño Index | El Niño / La Niña / Neutral state |
| `soil_profile` | ISRIC SoilGrids v2.0 | pH, texture, organic carbon, NPK, CEC |
| `soil_ocr` | EasyOCR + regex parser | Structured fields from a photographed soil-test report |
| `voice_intake` | OpenAI Whisper | Transcribed query in 99 languages |

All data sources are free and require no API keys.

## Live demo (sample query)

> *"I am Ramesh Singh, a wheat farmer in Bathinda, Punjab. My family has always planted on November 5. Given the current ENSO state, my historical climate, the projected climate for 2025–2035, and my soil profile — should I keep the November 5 date this year, or shift it?"*

Gemma 4 calls four tools in parallel, fetches real data, and recommends planting between **November 1–8** — citing the specific ONI value (-0.16°C, Neutral), soil pH (7.8, alkaline), and projected 2050 temperature delta (+1.22°C above current).

Full response: see [`outputs/day2_agent_response.txt`](outputs/day2_agent_response.txt).

## Notebooks

| Notebook | What it shows |
|----------|---------------|
| `01_baseline_climate_fingerprint` | Real Gemma 4 inference on Kaggle T4 + climate trend visualization |
| `02_function_calling_agent` | Gemma 4 chains four tool calls, synthesizes grounded answer |
| `03_multimodal_ocr_voice` | OCR-extracted soil report → Gemma; Whisper voice → Gemma |

## Engineering notes

- **Custom parser for Gemma 4 E4B's compact tool-call format** (`call:name{key:value}`). The format isn't documented anywhere we could find; we wrote it from scratch by observing the model's output.
- **Workaround for a Gemma 4 chat-template multi-turn replay bug** (`UndefinedError: 'dict object' has no attribute 'function'`). We flatten to a two-call architecture (planning → synthesis) instead of multi-turn assistant/tool replay.
- **Two upstream Gemma 4 multimodal bugs found and worked around** — vision (chat-template / image-processor token mismatch) and audio (`torch.finfo()` crash on quantized weights). Documented in [`docs/day3_vision_investigation.md`](docs/day3_vision_investigation.md). Pivoted to compose specialized open models (EasyOCR, Whisper) around Gemma's text core — same pattern production multimodal systems use.

## Repository structure

```
climate-calendar/
├── docs/                          architecture + investigation notes
├── notebooks/                     Day 1–3 Kaggle notebooks (run-all reproducible)
├── src/
│   ├── agent.py                   function-calling agent loop
│   └── tools/                     callable tools (climate, soil, ENSO, OCR, voice)
├── outputs/                       saved demo responses
├── PROGRESS.md                    daily build log
└── README.md
```

## Reproducing the demo

1. Open `notebooks/02_function_calling_agent.ipynb` in Kaggle (T4 GPU)
2. Run Cell 2 (install Transformers main) → **Factory Reset Session**
3. Run remaining cells top-to-bottom
4. ~90 seconds end-to-end; full agent loop with four tool calls and grounded synthesis

## License

Apache 2.0 (code) + CC-BY 4.0 (writeup, per hackathon rules)

## Built for the Gemma 4 Good Hackathon

[Hackathon page](https://kaggle.com/competitions/gemma-4-good-hackathon) · Submission deadline May 18, 2026
