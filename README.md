# Microsoft VibeVoice TTS Generator

Autonomous voiceover generation pipeline powered by GitHub Actions and Microsoft VibeVoice (`microsoft/VibeVoice-Realtime-0.5B`) & Microsoft Neural TTS.

## Workflow

This repository automatically generates narration voiceovers from `script.txt` on every push or manual `workflow_dispatch` trigger:

1. Downloads the Microsoft VibeVoice 0.5B parameter TTS model from Hugging Face.
2. Generates sentence-by-sentence voiceover with millisecond-accurate timestamps (`timestamps.json`).
3. Exports both high-fidelity lossless WAV (`vo.wav`) and compressed MP3 (`vo.mp3`).
4. Uploads all generated audio files as downloadable GitHub Actions artifacts (`voiceover-artifact`).
