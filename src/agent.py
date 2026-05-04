"""ClimateCalendar agent loop — Gemma 4 with native function calling.

Defines the tool catalog, dispatches Gemma's tool-call requests to the
Python implementations, and feeds results back into the model until it
produces a final answer.

Designed to run in a Kaggle notebook where `processor` and `model` are
already loaded.
"""

from typing import Any, Callable
import json
import re

from src.tools.climate_trend import get_climate_trend
from src.tools.climate_projection import get_climate_projection
from src.tools.enso import get_enso_state
from src.tools.soil_profile import get_soil_profile


# ---- Tool catalog: JSON-schema-style definitions Gemma 4 can call ----------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_climate_trend",
            "description": (
                "Fetch the historical 30-year climate trend (yearly mean "
                "temperature and total rainfall) for a GPS location. "
                "Use this to understand how the climate has already changed "
                "at the farmer's specific location."
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
                "Fetch the topsoil profile (texture, pH, organic carbon, CEC) for a "
                "GPS location from ISRIC SoilGrids. Use this when the farmer asks "
                "about fertilizer, soil amendments, or crop suitability."
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


# ---- Tool dispatcher --------------------------------------------------------

_DISPATCH: dict[str, Callable[..., Any]] = {
    "get_climate_trend": get_climate_trend,
    "get_climate_projection": get_climate_projection,
    "get_enso_state": get_enso_state,
    "get_soil_profile": get_soil_profile,
}


def execute_tool_call(name: str, arguments: dict) -> Any:
    """Run a tool by name with given arguments. Always returns JSON-serializable."""
    if name not in _DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _DISPATCH[name](**arguments)
    except Exception as e:
        return {"error": str(e), "tool": name}


# ---- Agent loop -------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert agronomist for smallholder farmers. You have tools that "
    "fetch real climate data, soil profiles, and ENSO state. Always call the "
    "tools you need before answering — never guess at numbers. When recommending "
    "a planting window, give a 5-7 day range. Cite the data you used. Speak "
    "plainly and respectfully. If the farmer's question can be answered with "
    "one tool call, call one. If it requires multiple, call them and reason "
    "across the results."
)


def run_agent(
    user_query: str,
    processor,
    model,
    tools: list[dict] = TOOLS,
    max_iterations: int = 5,
    max_new_tokens: int = 400,
    verbose: bool = True,
) -> str:
    """Run the agentic loop until Gemma produces a final answer."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for iteration in range(max_iterations):
        try:
            inputs = processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)
        except Exception as e:
            # If the chat template can't render the tool message format, fall back
            # to inlining the tool results into a plain user message and asking Gemma
            # to synthesize from there.
            if verbose:
                print(f"\n[chat template fell back due to: {type(e).__name__}]")
            return _synthesize_from_tool_results(
                base_messages=messages,
                processor=processor,
                model=model,
                tool_results=[],
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

        input_len = inputs["input_ids"].shape[-1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        response_text = processor.decode(out[0][input_len:], skip_special_tokens=True)

        if verbose:
            print(f"\n--- iteration {iteration + 1} raw response ---")
            print(response_text)
            print("--- end raw ---\n")

        tool_calls = _extract_tool_calls(response_text)

        if not tool_calls:
            return response_text.strip()

        if verbose:
            for c in tool_calls:
                print(f"  → tool call: {c['name']}({c.get('arguments', {})})")

        # Execute tools and collect results for the synthesis step
        tool_results: list[tuple[str, dict, object]] = []
        for call in tool_calls:
            result = execute_tool_call(call["name"], call.get("arguments", {}))
            if verbose:
                preview = json.dumps(result, default=str)[:200]
                print(
                    f"  ← tool result: {preview}{'...' if len(preview) == 200 else ''}"
                )
            tool_results.append((call["name"], call.get("arguments", {}), result))

        # Skip the multi-turn assistant/tool message replay (template doesn't like it).
        # Instead, synthesize directly: hand Gemma a fresh user turn with all results inlined.
        return _synthesize_from_tool_results(
            base_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            processor=processor,
            model=model,
            tool_results=tool_results,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )

    return "[Agent stopped: max iterations reached.]"


def _synthesize_from_tool_results(
    base_messages: list[dict],
    processor,
    model,
    tool_results: list[tuple[str, dict, object]] | None = None,
    max_new_tokens: int = 400,
    verbose: bool = False,
) -> str:
    """Hand Gemma all the tool results in one shot and ask for a final answer."""
    if tool_results is None:
        tool_results = []

    # Build a clean prompt with the tool outputs inlined
    tool_block_lines = ["Here are the results from the tools I called for you:\n"]
    for name, args, result in tool_results:
        result_json = json.dumps(result, default=str, indent=2)
        # Truncate huge results so we don't blow context
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

    synthesis_messages = base_messages + [
        {"role": "user", "content": follow_up},
    ]

    inputs = processor.apply_chat_template(
        synthesis_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def _split_top_level_commas(s: str) -> list[str]:
    """Split a string on commas that aren't inside nested braces/brackets."""
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
    """Parse Gemma E4B's compact arg format: 'key:value,key:value'.

    Values can be:
    - numbers (int or float, possibly negative)
    - bare strings (no quotes)
    - booleans (true/false)
    - empty (for tools that take no args)
    """
    if not args_str.strip():
        return {}

    arguments: dict = {}
    for pair in _split_top_level_commas(args_str):
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        key = key.strip().strip('"\'')
        value = value.strip().strip('"\'')

        # Type-coerce the value
        if value.lower() == "true":
            arguments[key] = True
        elif value.lower() == "false":
            arguments[key] = False
        elif value.lower() in ("null", "none"):
            arguments[key] = None
        else:
            # Try int, then float, then leave as string
            try:
                arguments[key] = int(value)
            except ValueError:
                try:
                    arguments[key] = float(value)
                except ValueError:
                    arguments[key] = value
    return arguments


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from Gemma 4's response.

    Gemma 4 E4B emits tool calls in a compact, non-JSON format:
        call:tool_name{key:value,key:value}call:another_tool{...}

    Arguments are not quoted: keys and string values appear bare.
    Numbers and floats appear as-is. We parse this format directly,
    while still supporting the standard ```tool_call``` JSON form
    as a fallback for the larger Gemma 4 sizes.
    """
    calls: list[dict] = []

    # Pattern 1 (E4B native format): call:tool_name{args}
    pattern = re.compile(r"call:([a-zA-Z_][a-zA-Z0-9_]*)\s*\{([^{}]*)\}")
    for m in pattern.finditer(text):
        tool_name = m.group(1)
        args_str = m.group(2).strip()
        arguments = _parse_compact_args(args_str)
        calls.append({"name": tool_name, "arguments": arguments})

    # Pattern 2 (fallback): ```tool_call\n{...}\n```
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

    # Pattern 3 (fallback): ```json {"name": "...", "arguments": {...}} ```
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
