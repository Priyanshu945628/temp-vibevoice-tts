#!/usr/bin/env python3
"""
tts.py - Microsoft VibeVoice & Microsoft Neural TTS Voiceover Generator
Generates narration voiceover audio (vo.wav / vo.mp3) and sentence timestamps (timestamps.json)
from script.txt.

Primary Model: microsoft/VibeVoice-Realtime-0.5B (Microsoft VibeVoice Realtime 0.5B TTS)
Voice Preset: Carter (en-Carter_man.pt)
Fallback: Microsoft Edge Neural TTS (en-US-ChristopherNeural / GuyNeural) for guaranteed resilience.
"""

import sys
import os
import re
import json
import copy
import math
import hashlib
import urllib.request
import subprocess
import numpy as np

DEFAULT_VOICE_NAME = "Carter"
VOICE_FILE_NAME = "en-Carter_man.pt"
VOICE_URL = f"https://raw.githubusercontent.com/microsoft/VibeVoice/1541f590c7099820f10ea012f48d2399282df69f/demo/voices/streaming_model/{VOICE_FILE_NAME}"
VOICE_SHA256 = "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897"
TOKEN_RATE_HZ = 7.5

def log(msg):
    print(f"[TTS] {msg}", flush=True)

def split_into_sentences(text):
    # Remove markdown headers and stage directions like **[0:00 - 0:25 | HOOK]**
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("---") or line.startswith("[END]"):
            continue
        if re.match(r'^\*?\*?\[\d+:\d+.*\]\*?\*?$', line):
            continue
        # Clean formatting asterisks
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        line = re.sub(r'\*(.*?)\*', r'\1', line)
        cleaned_lines.append(line)

    full_text = " ".join(cleaned_lines)
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', full_text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # Split sentences over ~180 chars at commas for smooth flow
        if len(s) > 180 and ',' in s:
            parts = re.split(r'(?<=,)\s+', s)
            current = []
            current_len = 0
            for part in parts:
                if current_len + len(part) > 180 and current:
                    sentences.append(' '.join(current).strip())
                    current = [part]
                    current_len = len(part)
                else:
                    current.append(part)
                    current_len += len(part)
            if current:
                sentences.append(' '.join(current).strip())
        else:
            sentences.append(s)
    return sentences

def calculate_max_new_tokens(num_chars, tighter=False):
    if tighter:
        max_sec = (num_chars / 11.0) + 1.0
    else:
        max_sec = (num_chars / 8.0) + 2.0
    return max(15, int(math.ceil(max_sec * TOKEN_RATE_HZ)))

def trim_trailing_silence(audio_np, sample_rate=24000, threshold=0.01, pad_ms=50):
    if len(audio_np) == 0:
        return audio_np
    abs_audio = np.abs(audio_np)
    max_amp = float(np.max(abs_audio)) if len(abs_audio) > 0 else 0.0
    eff_thresh = threshold
    if 0.0 < max_amp * 0.05 < eff_thresh:
        eff_thresh = max(0.001, max_amp * 0.05)
    active_indices = np.where(abs_audio > eff_thresh)[0]
    if len(active_indices) == 0:
        return audio_np
    last_idx = active_indices[-1]
    pad_samples = int(sample_rate * (pad_ms / 1000.0))
    end_idx = min(len(audio_np), last_idx + pad_samples)
    return audio_np[:end_idx]

def get_voice_preset_path():
    cache_base = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    voices_dir = os.path.join(cache_base, "voices")
    os.makedirs(voices_dir, exist_ok=True)
    voice_path = os.path.join(voices_dir, VOICE_FILE_NAME)

    if not os.path.exists(voice_path) or os.path.getsize(voice_path) == 0:
        log(f"Downloading Carter preset voice ({VOICE_FILE_NAME})...")
        try:
            req = urllib.request.Request(VOICE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(voice_path, "wb") as f:
                f.write(resp.read())
            log(f"Preset voice downloaded to {voice_path} ({os.path.getsize(voice_path)} bytes)")
        except Exception as e:
            log(f"Could not download voice preset: {e}")
            return None
    return voice_path

def run_vibevoice_tts(sentences, output_wav, timestamps_path):
    import torch
    import soundfile as sf
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    log("Loading Microsoft VibeVoice-Realtime-0.5B model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    attn_impl = "sdpa"
    model_id = "microsoft/VibeVoice-Realtime-0.5B"

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    voice_path = get_voice_preset_path()
    if not voice_path or not os.path.exists(voice_path):
        raise RuntimeError("Missing Carter voice preset file.")

    log(f"Loading speaker preset from {voice_path}...")
    try:
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            all_prefilled_outputs = torch.load(voice_path, map_location=device, weights_only=True)
    except Exception:
        all_prefilled_outputs = torch.load(voice_path, map_location=device, weights_only=False)

    sample_rate = 24000
    pause_samples = int(sample_rate * 0.15)
    pause_silence = np.zeros(pause_samples, dtype=np.float32)

    all_audio_chunks = []
    timestamps = []
    current_time_sec = 0.0

    log(f"Synthesizing {len(sentences)} sentence chunks...")
    for idx, sentence in enumerate(sentences):
        clean_text = sentence.replace("'", "'").replace('"', '"').replace('"', '"')
        num_chars = len(clean_text)

        def generate_chunk(tokens_bound):
            inputs = processor.process_input_with_cached_prompt(
                text=clean_text,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=tokens_bound,
                    cfg_scale=1.5,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
                )

            chunk_audio = outputs.speech_outputs[0]
            if torch.is_tensor(chunk_audio):
                return chunk_audio.squeeze().cpu().numpy().astype(np.float32)
            return np.array(chunk_audio, dtype=np.float32).squeeze()

        tokens = calculate_max_new_tokens(num_chars, tighter=False)
        raw_audio = generate_chunk(tokens)
        chunk_np = trim_trailing_silence(raw_audio, sample_rate=sample_rate, threshold=0.01)
        chunk_dur_sec = len(chunk_np) / float(sample_rate)

        start_sec = current_time_sec
        end_sec = start_sec + chunk_dur_sec
        timestamps.append({
            "index": idx,
            "text": sentence,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3)
        })

        all_audio_chunks.append(chunk_np)
        all_audio_chunks.append(pause_silence)
        current_time_sec = end_sec + 0.15
        log(f"Chunk {idx+1}/{len(sentences)} done: {chunk_dur_sec:.2f}s - {sentence[:40]}...")

    final_audio = np.concatenate(all_audio_chunks)
    sf.write(output_wav, final_audio, sample_rate, subtype="PCM_16")

    with open(timestamps_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=2)

    log(f"VibeVoice synthesis complete: {output_wav} ({current_time_sec:.2f}s total)")

def run_edge_tts(sentences, output_wav, timestamps_path):
    """Resilient fallback using Microsoft Edge Neural TTS."""
    log("Running Microsoft Edge Neural TTS (en-US-ChristopherNeural)...")
    import asyncio
    import edge_tts

    voice = "en-US-ChristopherNeural"
    temp_dir = "temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)

    chunk_files = []
    timestamps = []
    current_time_sec = 0.0

    async def gen_chunk(text, out_file):
        communicate = edge_tts.Communicate(text, voice, rate="+0%")
        await communicate.save(out_file)

    for idx, sentence in enumerate(sentences):
        chunk_file = os.path.join(temp_dir, f"chunk_{idx:03d}.mp3")
        asyncio.run(gen_chunk(sentence, chunk_file))

        # Check duration with ffprobe
        res = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", chunk_file
        ], capture_output=True, text=True)
        dur = float(res.stdout.strip()) if res.stdout.strip() else 2.0

        start_sec = current_time_sec
        end_sec = start_sec + dur
        timestamps.append({
            "index": idx,
            "text": sentence,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3)
        })
        current_time_sec = end_sec + 0.18
        chunk_files.append(chunk_file)
        log(f"Edge TTS Chunk {idx+1}/{len(sentences)}: {dur:.2f}s - {sentence[:40]}...")

    # Concatenate all chunks with ffmpeg into output_wav
    concat_list = os.path.join(temp_dir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for cf in chunk_files:
            f.write(f"file '{os.path.abspath(cf)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c:a", "pcm_s16le", "-ar", "24000", output_wav
    ], check=True)

    with open(timestamps_path, "w", encoding="utf-8") as f:
        json.dump(timestamps, f, indent=2)

    log(f"Edge TTS complete: {output_wav} ({current_time_sec:.2f}s)")

def main():
    script_path = sys.argv[1] if len(sys.argv) > 1 else "script.txt"
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!", file=sys.stderr)
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    sentences = split_into_sentences(script_content)
    log(f"Parsed {len(sentences)} sentences from {script_path}")

    os.makedirs("audio", exist_ok=True)
    output_wav = os.path.join("audio", "vo.wav")
    output_mp3 = os.path.join("audio", "vo.mp3")
    timestamps_path = os.path.join("audio", "timestamps.json")

    # Try VibeVoice first, fallback to Microsoft Edge TTS
    try:
        log("Attempting Microsoft VibeVoice-Realtime-0.5B...")
        run_vibevoice_tts(sentences, output_wav, timestamps_path)
    except Exception as e:
        log(f"VibeVoice failed or not installed ({e}). Falling back to Microsoft Neural TTS...")
        run_edge_tts(sentences, output_wav, timestamps_path)

    # Convert output WAV to MP3
    if os.path.exists(output_wav):
        subprocess.run([
            "ffmpeg", "-y", "-i", output_wav,
            "-codec:a", "libmp3lame", "-b:a", "192k", output_mp3
        ], check=True)
        log(f"Generated MP3: {output_mp3} ({os.path.getsize(output_mp3)} bytes)")

if __name__ == "__main__":
    main()
