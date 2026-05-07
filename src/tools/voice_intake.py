"""Voice intake — Whisper-based ASR for farmer voice notes.

Takes an audio file path, returns transcribed text and detected language.
Whisper handles 99 languages including Hindi, Punjabi, English.

Source: https://github.com/openai/whisper
Free, runs locally, no API key.
"""

from typing import Optional, TypedDict

import whisper


class VoiceTranscript(TypedDict):
    transcript: str
    detected_language: str
    audio_path: str
    explanation: str


# Lazy-loaded so we don't load weights unless transcribe is called
_whisper_model = None


def _get_model(model_size: str = "base", device: Optional[str] = None):
    """Load Whisper model on first use, cache afterwards."""
    global _whisper_model
    if _whisper_model is None:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        _whisper_model = whisper.load_model(model_size, device=device)
    return _whisper_model


def transcribe_voice(audio_path: str, model_size: str = "base") -> VoiceTranscript:
    """Transcribe a farmer's voice note to text.

    Args:
        audio_path: Path to an audio file (WAV, FLAC, MP3, M4A, OGG).
        model_size: Whisper model size. "base" is fast and good enough for
            most cases. Use "small" or "medium" for noisy audio or rare
            languages.

    Returns:
        Dict with transcript, detected language, and a one-line explanation.
    """
    model = _get_model(model_size)
    result = model.transcribe(audio_path)

    transcript = result["text"].strip()
    detected_language = result.get("language", "unknown")

    explanation = (
        f"Transcribed {len(transcript)} characters from {audio_path} "
        f"(detected language: {detected_language})."
    )

    return {
        "transcript": transcript,
        "detected_language": detected_language,
        "audio_path": audio_path,
        "explanation": explanation,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voice_intake.py <audio_file_path>")
        sys.exit(1)
    out = transcribe_voice(sys.argv[1])
    print(out["explanation"])
    print(f"Transcript: {out['transcript']}")
