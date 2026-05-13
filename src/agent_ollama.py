"""ClimateCalendar agent loop — Ollama backend with native function calling.

Adapted from src/agent.py to use Ollama HTTP API instead of transformers.
Preserves the two-call structure:
  1. Planning call: model sees query + tool catalog, emits tool calls
  2. Tool execution in Python
  3. Synthesis call: model sees query + tool results, writes final answer
"""

from typing import Any, Callable, Iterator
import json
import requests

from src.tools.climate_trend import get_climate_trend
from src.tools.climate_projection import get_climate_projection
from src.tools.enso import get_enso_state
from src.tools.soil_profile import get_soil_profile
from src.tools.viability_2050 import get_viability_projection

# Re-use the existing tool catalog and parser from agent.py
from src.agent import TOOLS, _extract_tool_calls, execute_tool_call


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "climate-calendar"

DEFAULT_LAT = 30.21   # Bathinda, Punjab
DEFAULT_LNG = 74.94

# Language steering — used in both planning and synthesis prompts.
# "auto" = let model detect from user input (no steering).
LANGUAGE_INSTRUCTIONS = {
    "auto": "",
    "english": "Respond in English.",
    "hindi": "Respond in Hindi (Devanagari script). Use clear Hindi for farmers.",
    "punjabi": "Respond in Punjabi (Gurmukhi script). Use clear Punjabi for farmers.",
}

# Steer the model toward PAU-recommended varieties when responding in any language.
VARIETY_STEERING = (
    "When recommending wheat varieties, use only PAU-recommended names: "
    "HD 3086, PBW 826, DBW 187, PBW 675, PBW 343, PBW 725. "
    "For rice: PR 126, PR 131, PR 130, Pusa Basmati 1509, 1718. "
    "For cotton: F 2228, RCH 650 BG II, NCS 855 BG II. "
    "For maize: PMH 1, PMH 2, PMH 10, JH 3459. "
    "Do not invent variety names like 'HBV-1' or 'PBW 1'."
)


def _build_tool_catalog_text(tools: list[dict]) -> str:
    """Format the tool catalog as a plain text block to inject into Ollama's prompt."""
    lines = ["You have access to the following tools:\n"]
    for t in tools:
        fn = t["function"]
        name = fn["name"]
        desc = fn["description"]
        params = fn["parameters"]["properties"]
        required = fn["parameters"].get("required", [])
        param_strs = []
        for p_name, p_info in params.items():
            req = " (required)" if p_name in required else ""
            param_strs.append(f"{p_name}: {p_info['type']}{req}")
        params_text = ", ".join(param_strs) if param_strs else "(no parameters)"
        lines.append(f"- {name}({params_text})\n  {desc}\n")
    return "\n".join(lines)


def _ollama_generate(prompt: str, stream: bool = False):
    """Call Ollama's /api/generate endpoint."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "think": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
            "num_predict": 700,
        },
    }

    if stream:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180)
        response.raise_for_status()

        def _stream_chunks():
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("response", "")
                if data.get("done"):
                    break

        return _stream_chunks()
    else:
        response = requests.post(OLLAMA_URL, json=payload, stream=False, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")


PLANNING_PROMPT_TEMPLATE = """You are an expert agronomist for Punjab smallholder farmers. You have tools that fetch real climate data, soil profiles, and ENSO state. Always call the tools you need before answering — never guess at numbers.

{tool_catalog}

When you need to call tools, emit each call on its own line in this EXACT format:
call:tool_name{{key1:value1,key2:value2}}

Examples:
call:get_climate_trend{{latitude:30.21,longitude:74.94,start_year:1995,end_year:2024}}
call:get_enso_state{{}}
call:get_viability_projection{{crop:wheat,latitude:30.21,longitude:74.94}}

Default location is Bathinda, Punjab: latitude {lat}, longitude {lng}.

When the farmer's question requires multiple tools, emit ALL the tool calls in your response, one per line. Do not write any other text — just the tool calls.

If the question is general agronomy advice that doesn't need real-time data (e.g., "what variety of wheat", "what is the ETL for whitefly"), respond directly without any tool calls. {language_instruction} {variety_steering}

Farmer's question: {query}

Your response (tool calls only, or direct answer):"""


SYNTHESIS_PROMPT_TEMPLATE = """You are an expert agronomist helping a Punjab smallholder farmer. The farmer asked you this question:

"{query}"

You called tools to get real data. Here are the results:

{tool_results}

Now write a final answer for the farmer using these real numbers. Requirements:
- Cite specific numbers from the data above (temperatures, ONI values, soil pH, etc.)
- If recommending planting, give a 5-7 day window
- Keep the answer under 250 words
- Speak plainly and respectfully to the farmer
- End with: "Source: PAU Package of Practices + live climate data."
- {language_instruction}
- {variety_steering}

Your answer:"""


def run_agent_streaming(
    user_query: str,
    lat: float = DEFAULT_LAT,
    lng: float = DEFAULT_LNG,
    language: str = "auto",
    verbose: bool = True,
) -> Iterator[tuple[str, str]]:
    """Run the agent and stream status + final answer to Gradio.

    Yields tuples of (status_message, partial_answer).
    """

    # ---- Step 1: planning call ----
    yield ("Planning...", "")

    tool_catalog = _build_tool_catalog_text(TOOLS)
    language_instruction = LANGUAGE_INSTRUCTIONS.get(language, "")
    planning_prompt = PLANNING_PROMPT_TEMPLATE.format(
        tool_catalog=tool_catalog,
        lat=lat,
        lng=lng,
        query=user_query,
        language_instruction=language_instruction,
        variety_steering=VARIETY_STEERING,
    )

    plan_text = _ollama_generate(planning_prompt, stream=False)

    if verbose:
        print("\n--- Planning response ---")
        print(plan_text)
        print("--- end planning ---\n")

    tool_calls = _extract_tool_calls(plan_text)

    # ---- If no tool calls, return the direct answer ----
    if not tool_calls:
        yield ("Answering directly...", "")
        full_answer = ""
        for chunk in _ollama_generate(user_query, stream=True):
            full_answer += chunk
            yield ("Answering directly...", full_answer)
        return

    # ---- Step 2: execute tools ----
    tool_results: list[tuple[str, dict, object]] = []
    for call in tool_calls:
        status = f"Calling {call['name']}..."
        yield (status, "")

        if verbose:
            print(f"  -> {call['name']}({call.get('arguments', {})})")

        result = execute_tool_call(call["name"], call.get("arguments", {}))

        if verbose:
            preview = json.dumps(result, default=str)[:200]
            print(f"  <- result: {preview}")

        tool_results.append((call["name"], call.get("arguments", {}), result))

    # ---- Step 3: synthesis call ----
    yield ("Synthesizing answer with real data...", "")

    tool_results_text_lines = []
    for name, args, result in tool_results:
        result_json = json.dumps(result, default=str, indent=2)
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "\n... [truncated]"
        tool_results_text_lines.append(f"### {name}({json.dumps(args)})\n{result_json}\n")
    tool_results_text = "\n".join(tool_results_text_lines)

    synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        query=user_query,
        tool_results=tool_results_text,
        language_instruction=language_instruction,
        variety_steering=VARIETY_STEERING,
    )

    full_answer = ""
    for chunk in _ollama_generate(synthesis_prompt, stream=True):
        full_answer += chunk
        yield ("Synthesizing answer with real data...", full_answer)
