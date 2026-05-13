"""ClimateCalendar Gradio Demo - Day 8 Phase 3
Multilingual (English / Hindi / Punjabi) agent UI with language dropdown +
auto-detect fallback. Backed by local Ollama for offline operation.
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from src.agent_ollama import run_agent_streaming


LANGUAGE_OPTIONS = {
    "Auto (detect from input)": "auto",
    "English": "english",
    "हिन्दी (Hindi)": "hindi",
    "ਪੰਜਾਬੀ (Punjabi)": "punjabi",
}


EXAMPLES_BY_LANGUAGE = {
    "auto": [
        "What has the climate done in Bathinda from 1995 to 2024?",
        "Will wheat still be viable in Bathinda in 2050?",
        "What variety of wheat should I sow in Bathinda this November?",
    ],
    "english": [
        "What variety of wheat should I sow in Bathinda this November?",
        "What is the ETL for cotton whitefly?",
        "Will wheat still be viable in Bathinda in 2050?",
    ],
    "hindi": [
        "बठिंडा में नवंबर में कौन सी गेहूं की किस्म बोनी चाहिए?",
        "कपास की सफेद मक्खी का ETL क्या है?",
        "क्या 2050 में बठिंडा में गेहूं उगाई जा सकेगी?",
    ],
    "punjabi": [
        "ਬਠਿੰਡਾ ਵਿੱਚ ਨਵੰਬਰ ਵਿੱਚ ਕਣਕ ਦੀ ਕਿਹੜੀ ਕਿਸਮ ਬੀਜਣੀ ਚਾਹੀਦੀ ਹੈ?",
        "ਨਰਮੇ ਦੀ ਚਿੱਟੀ ਮੱਖੀ ਦਾ ETL ਕੀ ਹੈ?",
        "ਕੀ 2050 ਵਿੱਚ ਬਠਿੰਡਾ ਵਿੱਚ ਕਣਕ ਉਗਾਈ ਜਾ ਸਕੇਗੀ?",
    ],
}


def respond(message, history, language_label):
    """Stream agent response with language steering."""
    if not message.strip():
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please ask a question."},
        ]
        return

    language_code = LANGUAGE_OPTIONS.get(language_label, "auto")
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    last_status = ""
    answer = ""

    try:
        for status, partial_answer in run_agent_streaming(
            message,
            language=language_code,
            verbose=True,
        ):
            if partial_answer:
                answer = partial_answer
                history[-1] = {
                    "role": "assistant",
                    "content": f"_[{status}]_\n\n{answer}",
                }
                yield history
            elif status != last_status:
                last_status = status
                history[-1] = {
                    "role": "assistant",
                    "content": f"_{status}_",
                }
                yield history
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"Error: {str(e)}"}
        yield history


def update_examples(language_label):
    """Refresh example queries when language changes."""
    code = LANGUAGE_OPTIONS.get(language_label, "auto")
    return gr.update(samples=[[ex] for ex in EXAMPLES_BY_LANGUAGE[code]])


with gr.Blocks(title="ClimateCalendar - Punjab Agronomy AI") as demo:
    gr.Markdown(
        """
        # ClimateCalendar — Punjab Agronomy AI

        **Climate-aware agronomy assistant for Punjab smallholder farmers.**

        Fine-tuned Gemma 4 E4B + 5 agent tools fetching live climate data
        (Open-Meteo ERA5, CMIP6 projections, NOAA ENSO, ISRIC SoilGrids).
        Running fully offline via Ollama on local hardware.

        **Default location: Bathinda, Punjab (30.21°N, 74.94°E)** &nbsp;&nbsp;|&nbsp;&nbsp;
        Supports **English, हिन्दी, ਪੰਜਾਬੀ**
        """
    )

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=list(LANGUAGE_OPTIONS.keys()),
            value="Auto (detect from input)",
            label="Response language",
            scale=1,
        )

    _chatbot_kw: dict = {
        "height": 500,
        "label": "Chat",
        "avatar_images": (None, None),
    }
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        _chatbot_kw["type"] = "messages"
    chatbot = gr.Chatbot(**_chatbot_kw)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question (English / हिन्दी / ਪੰਜਾਬੀ)...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    examples = gr.Dataset(
        samples=[[ex] for ex in EXAMPLES_BY_LANGUAGE["auto"]],
        components=[msg],
        label="Example questions:",
    )

    # Wiring
    msg.submit(respond, [msg, chatbot, language_dropdown], chatbot).then(
        lambda: "", outputs=msg
    )
    send_btn.click(respond, [msg, chatbot, language_dropdown], chatbot).then(
        lambda: "", outputs=msg
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

    language_dropdown.change(update_examples, language_dropdown, examples)
    examples.click(lambda x: x[0], examples, msg)

    gr.Markdown(
        """
        ---
        **Model:** [tkaushik015/climate-calendar-gemma4-e4b-lora](https://huggingface.co/tkaushik015/climate-calendar-gemma4-e4b-lora)
        &nbsp;|&nbsp;
        **Source:** [github.com/tkaushik015/climate-calendar](https://github.com/tkaushik015/climate-calendar)
        &nbsp;|&nbsp;
        Built for the Gemma 4 Good Hackathon, May 2026.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
