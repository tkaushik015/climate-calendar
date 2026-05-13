"""
ClimateCalendar Gradio Demo - Day 8 Phase 1 (Gradio 5)
Talk to the fine-tuned Gemma 4 E4B model via local Ollama.
"""
import gradio as gr
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "climate-calendar"


def query_climate_calendar(message, history):
    """Send a query to the local Ollama model and stream the response."""
    if not message.strip():
        yield "Please ask a question about Punjab agronomy."
        return

    payload = {
        "model": MODEL_NAME,
        "prompt": message,
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 2048,
            "num_predict": 384,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        partial_text = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            partial_text += token
            yield partial_text

            if data.get("done"):
                break
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to Ollama. Is it running? Try: `ollama serve`"
    except Exception as e:
        yield f"Error: {str(e)}"


EXAMPLES = [
    "What variety of wheat should I sow in Bathinda Punjab this November?",
    "What is the ETL for cotton whitefly?",
    "How much paddy straw is produced in Punjab annually?",
    "I am Ramesh from Bathinda, a smallholder wheat farmer. What is my biggest climate risk?",
    "Should I switch from rice to maize on water-stressed fields?",
    "How do I correct zinc deficiency in maize during the season?",
]


demo = gr.ChatInterface(
    fn=query_climate_calendar,
    title="ClimateCalendar - Punjab Agronomy AI",
    description=(
        "**Climate-aware agronomy assistant for Punjab smallholder farmers.**\n\n"
        "Fine-tuned Gemma 4 E4B trained on Punjab Agricultural University's Package of Practices. "
        "Running fully offline via Ollama on local hardware.\n\n"
        "Ask about wheat, rice, cotton, maize, climate adaptation, or specific Punjab varieties."
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
