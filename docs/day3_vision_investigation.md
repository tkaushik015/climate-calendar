# Day 3 Vision & Audio Investigation

## Status: Gemma 4 native multimodal blocked on upstream issues

We attempted to use Gemma 4 E4B's native vision and audio paths for the Day 3
multimodal demo. Both hit reproducible upstream bugs in the current Transformers
main branch (commit ce65b95..., May 4, 2026, on Kaggle). Pivoted to a composition
approach: PaddleOCR/EasyOCR for documents + OpenAI Whisper for voice + Gemma 4
text agent for reasoning.

## Bug 1: Vision — chat template / image processor mismatch

When loading Gemma 4 E4B in 4-bit quantization (BitsAndBytes nf4) and trying
to process an image:

- `processor(text, images, return_tensors="pt")` produces `pixel_values` with
  shape `[1, 2520, 768]` — multi-patch SigLIP feature embeddings, not raw pixels
- The chat template inserts only ~260 `<|image|>` placeholder tokens
- Encoder produces 2520 image features
- Mismatch causes `ValueError: Image features and image tokens do not match,
  tokens: 0/260, features: 260/2520` or "I see a blank gray image" hallucinations

### What we tried

- `apply_chat_template(...)` with multimodal messages → silent image drop
- Direct `processor(images=img, text=prompt)` → token/feature count mismatch
- `<|image|>` placeholder explicitly added → still mismatch
- `do_pan_and_scan=False` (kwarg form) → rejected by processor signature
- `images_kwargs={"do_pan_and_scan": False}` → TypeError
- `processor.image_processor.do_pan_and_scan = False` → silently ignored
- Image resized to 224x224 → tile count unchanged

## Bug 2: Audio — torch.finfo crashes on quantized weights

When loading Gemma 4 E4B in 4-bit and trying to transcribe an audio file:

- Crashes inside the audio tower at `modeling_gemma4.py:403`:
gradient_clipping = min(self.gradient_clipping,
torch.finfo(self.ffw_layer_1.linear.weight.dtype).max)
TypeError: torch.finfo() requires a floating point input type.
- The audio encoder calls `torch.finfo()` on weights that have been integer-
  quantized by BitsAndBytes (4-bit nf4). This is a real upstream bug in the
  Gemma 4 audio encoder when used with quantization.

## Decision

Composed specialized open models around Gemma 4's text core:

- **EasyOCR** for soil-test report extraction (regex parser handles OCR noise)
- **OpenAI Whisper-base** for voice transcription (99 languages including
  Hindi and Punjabi)
- **Gemma 4 E4B (text-only path)** for reasoning over OCR/voice outputs

This is the same composition pattern production multimodal systems use. We
lose the "Gemma sees the field" demo for now, but the multimodal experience
(point camera at soil report → app reads it → agent reasons over it; speak
in your language → app responds) is preserved.

## Reconsider when

- Transformers ships a stable Gemma 4 vision integration (likely days, since
  this is upstream)
- A non-quantized Gemma 4 deployment makes the audio path viable

For now we ship multimodal via composition. The writeup notes this honestly.
