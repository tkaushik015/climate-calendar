## Day 7 — May 10, 2026 — Fine-tune complete ✅

- Trained LoRA adapter on Gemma 4 E4B (4-bit) with **759** PAU agronomy examples
- **14-minute** training run on Kaggle T4 (rank 16, alpha 32, 2 epochs, batch 8, lr 2e-4)
- **42.4M** trainable params (**0.53%** of 8B total)
- Final train loss **0.668**, val loss **2.95** (mild overfit, acceptable)
- A/B comparison shows clear improvement:
  - Base Gemma 4: "ETL might be a typo" for cotton whitefly question
  - Fine-tuned: "ETL = Economic Threshold Level, 5 adult whiteflies per leaf"
  - Base: "consult your KVK" for variety question
  - Fine-tuned: names **HD 3086**, **PBW 671** specifically for late November sowing
  - Base: "I don't have data" for paddy straw
  - Fine-tuned: "1.5 crore tonnes annually" with 5-item usage list
- Adapter published: [tkaushik015/climate-calendar-gemma4-e4b-lora](https://huggingface.co/tkaushik015/climate-calendar-gemma4-e4b-lora)
- Local backup zip in `~/Downloads/climate_calendar_lora.zip`

## Day 2 (May 4, 2026) — Complete ✅

Built the function-calling agentic spine of ClimateCalendar.

### Tools added (`src/tools/`)
- **enso.py** — NOAA ONI scraper, current state classification (El Niño / La Niña / Neutral)
- **soil_profile.py** — ISRIC SoilGrids per-GPS profile with retry+cache+offset sweep for sparse tiles
- **climate_projection.py** — Open-Meteo CMIP6 climate projections to 2050

### Agent loop (`src/agent.py`)
- Two-step architecture (planning → tool execution → grounded synthesis)
- Custom regex parser for Gemma 4 E4B's compact `call:name{args}` tool-call format
- Workaround for a Gemma 4 chat-template multi-turn replay bug
  (`UndefinedError: 'dict object' has no attribute 'function'`):
  flatten to two single-turn Gemma calls instead of multi-turn assistant/tool replay
- System prompt instructs Gemma to emit ALL needed tool calls in a single response

### Live demo
- Ramesh Singh / Bathinda query: 4 tools called in parallel, real data fetched,
  grounded synthesis with cited numbers (ONI, pH, temperature trends)
- Total runtime: ~90 seconds on Kaggle T4 with 4-bit quantization

### Insights for the writeup
- Gemma 4 E4B's tool-call output format isn't well-documented; we wrote a
  parser for it from scratch
- The chat-template bug is a real implementation gotcha — the two-call
  architecture is honest engineering, not a workaround we should hide
- Honest reporting of missing data: when we tested without the strong
  system prompt, Gemma openly said "I do not have the specific ENSO state"
  rather than hallucinating. Good signal for the Safety & Trust track.

## Day 3 (May 5, 2026) — Complete ✅

ClimateCalendar now accepts three input modalities: text, document photos, voice.

### Tools added (`src/tools/`)
- **soil_ocr.py** — EasyOCR + regex parser for printed soil-test reports
  - Extracts pH, EC, organic carbon, NPK, zinc from a photographed report
  - Honest reporting of missing fields (low-confidence OCR results dropped)
- **voice_intake.py** — Whisper-base ASR wrapper
  - Transcribes farmer voice notes in any of 99 languages
  - Detected language returned alongside transcript

### Multimodal demo (notebook 03)
- OCR demo: synthesized realistic soil-test report → EasyOCR → robust regex
  parser → Gemma agent generates a grounded recommendation citing 5 of 7 OCR'd
  fields. Gemma honestly flags missing fields.
- Voice demo: Whisper transcribes English audio sample → Gemma agent
  processes the resulting query end-to-end.

### Engineering note
We tested Gemma 4 E4B's native vision and audio paths and hit two
upstream bugs:
1. Vision: chat-template / image-processor token-count mismatch
   (260 placeholders vs 2520 features); `do_pan_and_scan` kwargs are silently
   ignored. Result: Gemma sees a "blank gray image."
2. Audio: `torch.finfo()` crash inside the audio tower when model is loaded
   in 4-bit quantization. Real upstream bug at `modeling_gemma4.py:403`.

We pivoted to compose with specialized open models (EasyOCR + Whisper)
around Gemma 4's text core. This is documented in
`docs/day3_vision_investigation.md`. We will revisit native Gemma 4 vision
when upstream Transformers stabilizes the integration.

### Insights for the writeup
- Composing specialized open models around an LLM core is what production
  multimodal systems do — not a workaround, but a sound architecture.
- Honest field-extraction failures (OCR can't read Available P / K) → Gemma
  reasons over what's available and asks the farmer to re-photograph.
  Same Safety & Trust track signal as Day 2.
- Day 3 unblocks the demo video: farmer types/photographs/speaks → ClimateCalendar
  responds.
