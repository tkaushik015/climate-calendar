"""ClimateCalendar Gradio Demo - Day 8 Phase 2
Real agentic flow: model calls tools (climate, soil, ENSO, viability) and
synthesizes from live data. Backed by local Ollama for offline operation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.agent_ollama import run_agent_streaming


def respond(message, history):
    """Stream agent response to Gradio."""
    if not message.strip():
        yield "Please ask a question about Punjab agronomy."
        return

    answer = ""
    last_status = ""

    try:
        for status, partial_answer in run_agent_streaming(message, verbose=True):
            if partial_answer:
                answer = partial_answer
                yield f"_[{status}]_\n\n{answer}"
            elif status != last_status:
                last_status = status
                yield f"_{status}_"
    except Exception as e:
        yield f"Error: {str(e)}"


EXAMPLES = [
    "What has the climate done in Bathinda from 1995 to 2024?",
    "Will wheat still be viable in Bathinda in 2050?",
    "What is the current ENSO state and how does it affect Punjab agriculture?",
    "What is my soil like in Bathinda?",
    "What variety of wheat should I sow in Bathinda this November?",
    "What is the ETL for cotton whitefly?",
]


demo = gr.ChatInterface(
    fn=respond,
    title="ClimateCalendar - Punjab Agronomy AI",
    description=(
        "**Climate-aware agronomy assistant for Punjab smallholder farmers.**\n\n"
        "Fine-tuned Gemma 4 E4B + 5 agent tools fetching live climate data "
        "(Open-Meteo ERA5, CMIP6 projections, NOAA ENSO, ISRIC SoilGrids). "
        "Running fully offline via Ollama on local hardware.\n\n"
        "**Default location: Bathinda, Punjab (30.21°N, 74.94°E)**"
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
