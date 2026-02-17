# Voice Remix and Character Migration Engine

**Workflow 19** from [Advanced Workflows](../../ADVANCED_WORKFLOWS.md) — a full voice migration pipeline that swaps a speaker's voice in multi-speaker audio while preserving the original delivery, timing, and conversation structure.

## Source

This is a working implementation of **Workflow 19: Voice Remix and Character Migration Engine** as described in [`ADVANCED_WORKFLOWS.md`](../../ADVANCED_WORKFLOWS.md). The workflow was designed to demonstrate how multiple ElevenLabs APIs can be chained together to solve a real production problem: replacing a speaker's voice in existing audio content.

## What It Does

The pipeline takes multi-speaker audio and replaces one (or more) speakers' voices with a different voice identity, keeping everything else intact:

```
Multi-speaker audio
        |
  [Speech-to-Text + Diarization]  →  Identify WHO said WHAT and WHEN
        |
  [Speech-to-Speech]              →  Convert selected speaker's voice
        |
  [Timing Verification]           →  Ensure timing drift is acceptable
        |
  [Audio Reassembly]              →  Stitch remixed segments back in place
        |
  Final audio with swapped voice
```

### Pipeline Steps

1. **Generate test audio** (optional) — Creates a two-speaker conversation using TTS with Rachel and Adam voices
2. **Analyze source audio** — Transcribes with Scribe v2 + diarization to identify speakers and word-level timestamps
3. **Remix voice** — Sends each target speaker's segments through Speech-to-Speech to change the voice identity
4. **Verify timing** — Compares original vs remixed segment durations to detect drift (< 15% is acceptable)
5. **Reassemble** — Overlays remixed segments at their original timestamps, trimming or padding to fit the original slots

## ElevenLabs APIs and Features Used

| API / Feature | Skill | How It's Used |
|---------------|-------|---------------|
| **Text-to-Speech** (`client.text_to_speech.convert`) | [text-to-speech](../../text-to-speech/SKILL.md) | Generates the test conversation with two distinct voices (Rachel, Adam) using `eleven_multilingual_v2` model |
| **Speech-to-Text** (`client.speech_to_text.convert`) | [speech-to-text](../../speech-to-text/SKILL.md) | Transcribes the audio using Scribe v2 with `diarize=True` and `timestamps_granularity="word"` to get per-word speaker labels and millisecond timestamps |
| **Speech-to-Speech** (`client.speech_to_speech.convert`) | — | Converts each segment's voice identity to a target voice using `eleven_english_sts_v2` model while preserving the original prosody, pacing, and delivery |
| **Voice Settings** (`VoiceSettings`) | [text-to-speech](../../text-to-speech/SKILL.md) | Controls stability (0.6), similarity boost (0.75), and style (0.0) for natural-sounding TTS output |
| **Speaker Diarization** | [speech-to-text](../../speech-to-text/SKILL.md) | Automatically labels each word with a `speaker_id` so the pipeline knows which segments belong to which speaker |
| **Pre-built Voice Library** | [text-to-speech](../../text-to-speech/SKILL.md) | Uses well-known ElevenLabs voices by ID (Rachel, Adam, Josh, Bella, George) |

## Test Results

The pipeline was tested successfully with these results:

- **Input**: 24.4s generated conversation (6 lines, 2 speakers)
- **Diarization**: 6 segments correctly identified across 2 speakers
- **Remixed**: 3 segments (speaker_0 / Rachel → George)
- **Timing drift**: 0.9%, 1.6%, 1.3% — all well within the 15% tolerance
- **Output**: Seamless conversation with Rachel's lines now spoken in George's British male voice

```
Segment 0: 4219ms → 4258ms (0.9% drift) ✓
Segment 2: 3420ms → 3474ms (1.6% drift) ✓
Segment 4: 3741ms → 3788ms (1.3% drift) ✓
```

## Usage

### Prerequisites

```bash
pip install elevenlabs pydub python-dotenv
# ffmpeg must be installed and available on PATH
```

### Setup

Create a `.env` file in this directory:

```
ELEVENLABS_API_KEY=your_api_key_here
```

### Run

```bash
# Full pipeline with auto-generated test audio (two TTS speakers)
python voice_migration.py

# Use your own multi-speaker audio file
python voice_migration.py path/to/audio.mp3
```

### Customize

Edit the `speaker_voice_map` in `__main__` to remap different speakers to different voices:

```python
result = migrate_voices(
    audio_path=audio_path,
    speaker_voice_map={
        "speaker_0": VOICES["george"],   # Remap first speaker to George
        "speaker_1": VOICES["bella"],    # Remap second speaker to Bella
    },
)
```

## File Structure

```
playground/voice-migration/
├── voice_migration.py          # Full pipeline implementation
├── .env                        # API key (gitignored)
├── .gitignore                  # Excludes .env and output MP3s
├── README.md                   # This file
└── output/
    ├── test_conversation.mp3   # Generated test audio (gitignored)
    ├── migrated_conversation.mp3  # Output with swapped voice (gitignored)
    └── migration_report.json   # Timing and segment details
```
