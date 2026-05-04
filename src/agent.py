"""ClimateCalendar agent loop — Gemma 4 with native function calling.

Strategy: two Gemma calls per query.
  1. Call 1: Gemma sees the user query + tool catalog, emits tool calls.
  2. Python executes the tools.
  3. Call 2: Gemma sees the original query + inlined tool results, writes a
     grounded final answer.

This avoids Gemma 4 E4B's chat-template bug with multi-turn assistant/tool
messages, while still being a real agentic flow (model picks the tools,
tools run, model synthesizes from real data).
"""

from typing import Any, Callable
import json
import re

from src.tools.climate_trend import get_climate_trend
from src.tools.climate_projection import get_climate_projection
from src.tools.enso import get_enso_state
from src.tools.soil_profile import get_soil_profile


# ---- Tool catalog ----------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_climate_trend",
            "description": (
                "Fetch the historical climate trend (yearly mean temperature and "
                "total rainfall) for a GPS location. Use this to understand how "
                "the climate has already changed at the farmer's specific location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                },
                "required": ["latitude", "longitude", "start_year", "end_year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_climate_projection",
            "description": (
                "Fetch downscaled CMIP6 climate projections (yearly temperature "
                "and rainfall) for a GPS location, for any year range up to 2050. "
                "Use this to reason about future viability of crops or planting windows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                },
                "required": ["latitude", "longitude", "start_year", "end_year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_enso_state",
            "description": (
                "Get the current state of the El Niño-Southern Oscillation (ENSO) "
                "from NOAA's Oceanic Niño Index. Returns whether we are currently "
                "in El Niño, La Niña, or Neutral conditions, with intensity."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_soil_profile",
            "description": (
                "Fetch the topsoil profile (texture, pH, organic carbon, CEC) for "
                "a GPS location from ISRIC SoilGrids. Use this when the farmer "
                "asks about fertilizer, soil amendments, or crop suitability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]


# ---- Tool dispatch ---------------------------------------------------------

_DISPATCH: dict[str, Callable[..., Any]] = {
    "get_climate_trend": get_climate_trend,
    "get_climate_projection": get_climate_projection,
    "get_enso_state": get_enso_state,
    "get_soil_profile": get_soil_profile,
}


def execute_tool_call(name: str, arguments: dict) -> Any:
    """Run a tool by name, return JSON-serializable result."""
    if name not in _DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _DISPATCH[name](**arguments)
    except Exception as e:
        return {"error": str(e), "tool": name}


# ---- Tool-call parser (handles Gemma 4 E4B's compact format) ---------------

def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas not inside nested braces/brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _parse_compact_args(args_str: str) -> dict:
    """Parse Gemma E4B's 'key:value,key:value' (unquoted) format."""
    if not args_str.strip():
        return {}
    arguments: dict = {}
    for pair in _split_top_level_commas(args_str):
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        key = key.strip().strip('"\'')
        value = value.strip().strip('"\'')
        if value.lower() == "true":
            arguments[key] = True
        elif value.lower() == "false":
            arguments[key] = False
        elif value.lower() in ("null", "none"):
            arguments[key] = None
        else:
            try:
                arguments[key] = int(value)
            except ValueError:
                try:
                    arguments[key] = float(value)
                except ValueError:
                    arguments[key] = value
    return arguments


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from Gemma 4's response in any common format."""
    calls: list[dict] = []

    # Gemma 4 E4B native: call:tool_name{args}
    for m in re.finditer(r"call:([a-zA-Z_][a-zA-Z0-9_]*)\s*\{([^{}]*)\}", text):
        calls.append(
            {
                "name": m.group(1),
                "arguments": _parse_compact_args(m.group(2).strip()),
            }
        )

    # Fallback: ```tool_call ...```
    if not calls:
        for m in re.finditer(r"```tool_call\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                obj = json.loads(m.group(1))
                calls.append(
                    {
                        "name": obj.get("name")
                        or obj.get("function", {}).get("name"),
                        "arguments": obj.get("arguments")
                        or obj.get("function", {}).get("arguments", {}),
                    }
                )
            except json.JSONDecodeError:
                continue

    # Fallback: ```json {...}```
    if not calls:
        for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                obj = json.loads(m.group(1))
                if "name" in obj:
                    calls.append(
                        {
                            "name": obj["name"],
                            "arguments": obj.get("arguments", {}),
                        }
                    )
            except json.JSONDecodeError:
                continue

    known = {t["function"]["name"] for t in TOOLS}
    return [c for c in calls if c.get("name") in known]


# ---- Agent loop ------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert agronomist for smallholder farmers. You have tools that "
    "fetch real climate data, soil profiles, and ENSO state. Always call the "
    "tools you need before answering — never guess at numbers.\n\n"
    "IMPORTANT: When the farmer's question requires multiple tools, emit ALL "
    "the tool calls in your FIRST response, concatenated together. Do not "
    "describe what you will do — just call the tools. The system will execute "
    "them all in parallel and feed every result back to you. Then you can "
    "write your final answer grounded in all the data.\n\n"
    "When recommending a planting window, give a 5-7 day range. Cite specific "
    "numbers in your final answer. Speak plainly and respectfully."
)


def run_agent(
    user_query: str,
    processor,
    model,
    tools: list[dict] = TOOLS,
    max_new_tokens: int = 400,
    verbose: bool = True,
) -> str:
    """Run the agentic flow: tool selection → execution → grounded synthesis."""

    # ---- Step 1: ask Gemma which tools to call ----
    plan_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    plan_inputs = processor.apply_chat_template(
        plan_messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    plan_input_len = plan_inputs["input_ids"].shape[-1]
    plan_out = model.generate(
        **plan_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    plan_text = processor.decode(
        plan_out[0][plan_input_len:],
        skip_special_tokens=True,
    )

    if verbose:
        print("\n--- Step 1: tool selection raw response ---")
        print(plan_text)
        print("--- end raw ---\n")

    tool_calls = _extract_tool_calls(plan_text)

    # If Gemma didn't call any tools, return its direct answer
    if not tool_calls:
        if verbose:
            print("[no tool calls extracted — returning direct answer]")
        return plan_text.strip()

    # ---- Step 2: execute tools ----
    tool_results: list[tuple[str, dict, object]] = []
    for call in tool_calls:
        if verbose:
            print(f"  → tool call: {call['name']}({call.get('arguments', {})})")
        result = execute_tool_call(call["name"], call.get("arguments", {}))
        if verbose:
            preview = json.dumps(result, default=str)[:200]
            print(
                f"  ← tool result: {preview}{'...' if len(preview) == 200 else ''}"
            )
        tool_results.append((call["name"], call.get("arguments", {}), result))

    # ---- Step 3: hand results back to Gemma for grounded synthesis ----
    tool_block_lines = ["Here are the results from the tools I called for you:\n"]
    for name, args, result in tool_results:
        result_json = json.dumps(result, default=str, indent=2)
        if len(result_json) > 4000:
            result_json = result_json[:4000] + "\n... [truncated]"
        tool_block_lines.append(f"### {name}({json.dumps(args)})\n{result_json}\n")
    tool_block = "\n".join(tool_block_lines)

    follow_up = (
        f"{tool_block}\n"
        "Now write a final answer for the farmer. Cite specific numbers from "
        "the data above (ONI, temperature change, soil pH, etc.). Give a "
        "5-7 day planting window. Keep the answer under 250 words and speak "
        "plainly to the farmer."
    )

    synth_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "user", "content": follow_up},
    ]
    synth_inputs = processor.apply_chat_template(
        synth_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    synth_input_len = synth_inputs["input_ids"].shape[-1]
    synth_out = model.generate(
        **synth_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(
        synth_out[0][synth_input_len:],
        skip_special_tokens=True,
    ).strip()
