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
