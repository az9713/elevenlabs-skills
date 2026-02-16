# ElevenLabs Skills Guide

**What each skill does, why you need it, how to use it, and whether it follows best practices.**

This document covers every Agent Skill in the `elevenlabs/skills` repository. It is written for developers who want to integrate ElevenLabs audio AI into their applications and for contributors who want to understand (or improve) how the skills are structured.

## Quick Reference

| # | Skill | Path | SKILL.md | Refs | Scripts | Primary Use Case |
|---|-------|------|----------|------|---------|------------------|
| 1 | setup-api-key | `setup-api-key/` | 56 lines | 0 | 0 | Obtain and configure an API key |
| 2 | text-to-speech | `text-to-speech/` | 218 lines | 3 | 0 | Convert text to spoken audio |
| 3 | speech-to-text | `speech-to-text/` | 259 lines | 6 | 0 | Transcribe audio/video to text |
| 4 | sound-effects | `sound-effects/` | 120 lines | 1 | 0 | Generate sound effects from prompts |
| 5 | music | `music/` | 105 lines | 2 | 0 | Generate music from prompts |
| 6 | agents | `agents/` | 266 lines | 5 | 0 | Build conversational voice AI agents |
| 7 | elevenlabs-transcribe | `openclaw/elevenlabs-transcribe/` | 147 lines | 0 | 2 | CLI transcription (batch + realtime) |

## Prerequisites

- **API key**: All skills require `ELEVENLABS_API_KEY`. Run the `setup-api-key` skill first or create a key at <https://elevenlabs.io/app/settings/api-keys>.
- **Python SDK**: `pip install elevenlabs`
- **JavaScript SDK**: `npm install @elevenlabs/elevenlabs-js`
- **Install skills**: `npx skills add elevenlabs/skills`

> **JS SDK warning**: Always use `@elevenlabs/elevenlabs-js`. The bare `elevenlabs` npm package is a deprecated v1.x release that will not work with current examples.

---

# Part 1: Individual Skill Deep Dives

---

## 1. setup-api-key

### 1.1 WHAT -- Capabilities

A guided, interactive workflow that walks a user through obtaining, validating, and storing an ElevenLabs API key. It is the bootstrap skill: every other skill depends on the key it produces.

**Features:**
- Links directly to the sign-up and API-key pages
- Validates the key with a `GET /v1/user` call
- Saves the key to a `.env` file as `ELEVENLABS_API_KEY`
- Retries once on invalid key, then exits with an error

**File inventory:**
- `setup-api-key/SKILL.md` -- 56 lines
- No references, no scripts

### 1.2 WHY -- Problem Space

**Problems it solves:**
- New users do not know where to find their API key
- Invalid or expired keys cause cryptic 401 errors downstream
- Storing the key in the environment prevents hard-coding secrets

**When to use it:**
- First interaction with any ElevenLabs skill
- When API calls fail with 401 errors
- When a user mentions needing ElevenLabs access

**When NOT to use it:**
- The user already has a working `ELEVENLABS_API_KEY` in their environment
- You need to rotate or revoke keys (do that in the dashboard)

### 1.3 HOW -- Usage

The skill is an interactive agent script, not a library. The workflow is:

1. Present the user with the API-key page URL
2. Wait for the user to paste their key
3. Validate via `GET https://api.elevenlabs.io/v1/user` with header `xi-api-key: <key>`
4. On success, write `ELEVENLABS_API_KEY=<key>` to `.env`
5. On failure, prompt once more, then abort

**Key configuration**: None -- the skill has no parameters.

**Error handling**: Two validation attempts before exit.

### 1.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `setup-api-key` -- kebab-case, 13 chars, matches directory |
| `description` field | PASS | 213 chars, includes WHAT and trigger phrases |
| Progressive disclosure | PASS | 56 lines, well under the 500-line guideline |
| Content quality | PARTIAL | Clear workflow, but no `.gitignore` guidance and no handling of a pre-existing `.env` file |
| Security | PASS | No XML in YAML, no prohibited prefixes |
| Reference structure | PASS | No references needed for this size |
| **Spec compliance** | PARTIAL | Missing `metadata` field that all other official skills include |

**Specific findings:**
1. No `metadata` field in frontmatter (every other official skill has one)
2. No guidance on adding `.env` to `.gitignore` -- risk of committing secrets
3. Does not check for or merge with an existing `.env` file

---

## 2. text-to-speech

### 2.1 WHAT -- Capabilities

Converts text to natural-sounding speech using ElevenLabs voice AI. Supports 74+ languages, five model tiers from highest quality to ultra-low latency, configurable voice settings, and multiple output formats.

**Features:**
- Five models (`eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5`, `eleven_flash_v2`, `eleven_turbo_v2_5`)
- Voice settings: stability, similarity boost, style, speaker boost, speed
- Language enforcement via ISO 639-1 codes
- Text normalization control (auto/on/off)
- Request stitching for multi-part audio
- 7 output formats (MP3, PCM at various rates, u-law)
- HTTP streaming and WebSocket streaming
- Cost tracking via response headers

**File inventory:**
- `text-to-speech/SKILL.md` -- 218 lines
- `references/installation.md` -- 90 lines
- `references/streaming.md` -- 307 lines
- `references/voice-settings.md` -- 115 lines

### 2.2 WHY -- Problem Space

**Problems it solves:**
- Generating voiceovers, audiobooks, podcasts, or any spoken audio from text
- Building real-time voice applications (chatbots, IVR, game characters)
- Producing multilingual audio without hiring voice talent

**When to use it:**
- "Generate audio from text"
- "Create a voiceover"
- "Build a voice app"
- "Synthesize speech"
- "Text to audio"

**When NOT to use it:**
- You need to transcribe audio to text (use `speech-to-text`)
- You need non-speech audio like explosions or rain (use `sound-effects`)
- You need music (use `music`)

### 2.3 HOW -- Usage

**Minimal example (Python):**

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()
audio = client.text_to_speech.convert(
    text="Hello, welcome to ElevenLabs!",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2"
)
with open("output.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)
```

**Key configuration options:**
- `voice_id` -- choose a pre-made voice or clone your own
- `model_id` -- trade off quality vs. latency
- `voice_settings` -- fine-tune stability, similarity, style
- `output_format` -- `mp3_44100_128` (default), `pcm_24000`, `ulaw_8000`, etc.
- `language_code` -- force pronunciation language
- `apply_text_normalization` -- control how numbers/dates are spoken
- `previous_text` / `next_text` -- request stitching for seamless multi-part audio

**Advanced patterns:**
- WebSocket streaming for LLM-to-speech pipelines (send text chunks as they arrive from the LLM; documented in `references/streaming.md`)
- Use `flush: true` at sentence boundaries to force immediate audio generation
- Use `chunk_length_schedule` to control the quality/latency trade-off

**Error handling:**
- 401: Invalid API key
- 422: Invalid parameters (bad voice_id, bad model_id)
- 429: Rate limit exceeded

### 2.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `text-to-speech` -- kebab-case, 14 chars, matches directory |
| `description` field | PASS | 154 chars, WHAT + WHEN with trigger phrases |
| Progressive disclosure | PASS | 218-line SKILL.md with 3 focused reference files |
| Content quality | PASS | Python/JS/cURL examples, voice settings, streaming, error handling, cost tracking |
| Security | PASS | Clean YAML, no issues |
| Reference structure | PASS | Three focused files: installation, streaming, voice-settings |

---

## 3. speech-to-text

### 3.1 WHAT -- Capabilities

Transcribes audio and video files to text using the Scribe v2 model. Supports 90+ languages, speaker diarization (up to 32 speakers), word-level timestamps, keyterm prompting, audio event tagging, entity detection, and real-time streaming.

**Features:**
- Batch transcription (`scribe_v2`) and real-time streaming (`scribe_v2_realtime`, ~150ms latency)
- Speaker diarization with configurable sensitivity
- Keyterm prompting (up to 100 terms for domain-specific accuracy)
- Audio event tagging (laughter, applause, music)
- Entity detection (PII, PHI, PCI, offensive language)
- Multi-channel audio support
- Cloud storage URL transcription
- Webhook-based async processing
- Client-side streaming with React hooks (`useScribe`)
- Server-side streaming with manual commit or VAD auto-commit

**Supported formats:** MP3, WAV, M4A, FLAC, OGG, WebM, AAC, AIFF, Opus, MP4, AVI, MKV, MOV, WMV, FLV, MPEG, 3GPP (up to 3 GB, 10 hours)

**File inventory:**
- `speech-to-text/SKILL.md` -- 259 lines
- `references/installation.md` -- 92 lines
- `references/transcription-options.md` -- 174 lines
- `references/realtime-client-side.md` -- 169 lines
- `references/realtime-server-side.md` -- 316 lines
- `references/realtime-commit-strategies.md` -- 124 lines
- `references/realtime-events.md` -- 195 lines

### 3.2 WHY -- Problem Space

**Problems it solves:**
- Transcribing meetings, interviews, podcasts
- Generating subtitles and captions
- Building live transcription features
- Processing spoken content at scale

**When to use it:**
- "Transcribe this audio"
- "Convert speech to text"
- "Generate subtitles"
- "Transcribe a meeting"

**When NOT to use it:**
- You want a ready-made CLI tool (use `openclaw/elevenlabs-transcribe` instead)
- You want to generate speech from text (use `text-to-speech`)

### 3.3 HOW -- Usage

**Minimal example (Python):**

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()
with open("audio.mp3", "rb") as f:
    result = client.speech_to_text.convert(file=f, model_id="scribe_v2")
print(result.text)
```

**Key configuration options:**
- `model_id` -- `scribe_v2` (batch, highest accuracy) or `scribe_v2_realtime` (low latency)
- `diarize` -- enable speaker identification
- `timestamps_granularity` -- `none`, `word`, or `character`
- `keyterms` -- array of up to 100 domain-specific terms
- `language_code` -- ISO 639-1 or 639-3 language hint
- `tag_audio_events` -- detect non-speech sounds
- `entity_detection` -- detect PII/PHI/PCI/offensive language

**Advanced patterns:**
- Real-time streaming with VAD auto-commit for live microphone input
- Manual commit strategy for file processing with controlled segment boundaries
- Client-side React hook (`useScribe`) with single-use token authentication
- Providing `previous_text` context for better accuracy across reconnections

**Error handling:**
- 401: Invalid API key
- 422: Invalid parameters
- 429: Rate limit exceeded
- WebSocket-specific: `authentication_failed`, `quota_exceeded`, `invalid_audio`, `session_time_limit_exceeded`

### 3.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `speech-to-text` -- kebab-case, 14 chars, matches directory |
| `description` field | PASS | 141 chars, WHAT + WHEN with trigger phrases |
| Progressive disclosure | PASS | 259-line SKILL.md with 6 focused reference files |
| Content quality | PARTIAL | Excellent coverage, but has an unclosed code block (see below) |
| Security | PASS | Clean YAML |
| Reference structure | PASS | Six well-organized reference files for different streaming scenarios |

**Specific finding:**
- **BUG**: `speech-to-text/SKILL.md` line 157 -- missing closing ` ``` ` before the `## Real-Time Streaming` heading. The `## Tracking Costs` code block starting at line 157 is never closed, meaning the entire Real-Time Streaming section is technically inside a code fence. Renderers may still display it correctly, but it is syntactically invalid markdown.

---

## 4. sound-effects

### 4.1 WHAT -- Capabilities

Generates sound effects from text descriptions. Supports custom duration, prompt adherence control, seamless looping, and 21 output formats.

**Features:**
- Text-to-sound-effect generation with the `eleven_text_to_sound_v2` model
- Duration control: 0.5--30 seconds (or auto)
- Prompt influence: 0.0--1.0 (how closely to follow the description)
- Seamless loop generation (v2 model)
- 21 output formats including MP3, PCM, Opus, u-law, and a-law at various sample rates

**File inventory:**
- `sound-effects/SKILL.md` -- 120 lines
- `references/installation.md` -- 63 lines

### 4.2 WHY -- Problem Space

**Problems it solves:**
- Creating sound effects for games, videos, podcasts, or apps
- Generating ambient textures, UI sounds, cinematic impacts
- Producing looping background audio

**When to use it:**
- "Generate a sound effect"
- "Create ambient audio"
- "Make a notification chime"
- Any non-speech, non-music audio

**When NOT to use it:**
- You need spoken audio (use `text-to-speech`)
- You need music with melody, rhythm, or lyrics (use `music`)

### 4.3 HOW -- Usage

**Minimal example (Python):**

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()
audio = client.text_to_sound_effects.convert(
    text="Thunder rumbling in the distance with light rain",
)
with open("thunder.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)
```

**Key configuration options:**
- `text` (required) -- descriptive prompt for the desired sound
- `duration_seconds` -- 0.5 to 30s, or null for auto
- `prompt_influence` -- 0.0 to 1.0 (default 0.3), higher = stricter adherence
- `loop` -- boolean, generates a seamlessly looping sound (v2 only)
- `output_format` -- one of 21 formats (default `mp3_44100_128`)

**Prompt tips (from skill):**
- Be specific: "Heavy rain on a tin roof" beats "Rain"
- Combine elements: "Footsteps on gravel with distant traffic"
- Specify style: "Cinematic braam, horror" or "8-bit retro jump sound"
- Mention mood/context: "Eerie wind howling through an abandoned building"

**Error handling:**
- 401: Invalid API key
- 422: Invalid parameters (check duration range 0.5--30, prompt_influence range 0--1)
- 429: Rate limit exceeded

### 4.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `sound-effects` -- kebab-case, 13 chars, matches directory |
| `description` field | PASS | 206 chars, WHAT + WHEN with detailed trigger phrases |
| Progressive disclosure | PASS | 120 lines, concise and complete |
| Content quality | PASS | Parameters table, prompt tips, output formats, error handling |
| Security | PASS | Clean YAML |
| Reference structure | PASS | Single installation reference -- appropriate for scope |

---

## 5. music

### 5.1 WHAT -- Capabilities

Generates music from text prompts. Supports instrumental tracks, songs with lyrics, and fine-grained control via composition plans. Duration range 3--600 seconds.

**Features:**
- Three methods: `music.compose`, `music.composition_plan.create`, `music.compose_detailed`
- Prompt-based generation for quick results
- Composition plans for granular control over styles, sections, and lyrics
- `force_instrumental` flag to guarantee no vocals
- Content filtering with suggested alternatives for copyrighted material
- Detailed output with metadata and composition plan

**File inventory:**
- `music/SKILL.md` -- 105 lines
- `references/installation.md` -- 65 lines
- `references/api_reference.md` -- 165 lines

### 5.2 WHY -- Problem Space

**Problems it solves:**
- Creating background music for videos, games, or apps
- Generating jingles, intros, and outros
- Composing custom tracks without music production skills
- Producing royalty-free music on demand

**When to use it:**
- "Generate music"
- "Create a beat"
- "Make a jingle"
- "Compose a track"

**When NOT to use it:**
- You need sound effects without melody (use `sound-effects`)
- You need spoken audio (use `text-to-speech`)

### 5.3 HOW -- Usage

**Minimal example (Python):**

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()
audio = client.music.compose(
    prompt="A chill lo-fi hip hop beat with jazzy piano chords",
    music_length_ms=30000
)
with open("output.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)
```

**Key configuration options:**
- `prompt` -- text description of desired music
- `music_length_ms` -- duration in milliseconds (3,000--600,000)
- `composition_plan` -- alternative to prompt for granular control
- `force_instrumental` -- boolean, guarantees no vocals
- `respect_sections_durations` -- enforce exact section durations from plan

**Advanced pattern -- composition plans:**

```python
plan = client.music.composition_plan.create(
    prompt="An epic orchestral piece building to a climax",
    music_length_ms=60000
)
# Inspect and modify styles, sections, lyrics
audio = client.music.compose(composition_plan=plan, music_length_ms=60000)
```

**Error handling:**
- 401: Invalid API key
- 422: Invalid parameters
- 429: Rate limit exceeded
- `bad_prompt`: Copyrighted material detected -- response includes `prompt_suggestion`
- `bad_composition_plan`: Copyrighted styles detected -- response includes `composition_plan_suggestion`

### 5.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `music` -- kebab-case, 5 chars, matches directory |
| `description` field | PASS | 203 chars, WHAT + WHEN with trigger phrases |
| Progressive disclosure | PASS | 105 lines in SKILL.md, API details in reference |
| Content quality | PARTIAL | Missing output format documentation and prompt tips (see below) |
| Security | PASS | Clean YAML |
| Reference structure | PASS | Two references: installation + API reference |

**Specific findings:**
1. **Missing output formats**: `sound-effects` documents 21 output formats; `music` documents zero. Users have no idea what formats are available or how to request them.
2. **Missing prompt tips**: `sound-effects` has a helpful "Prompt Tips" section; `music` has none. Users would benefit from guidance on writing effective music prompts.

---

## 6. agents

### 6.1 WHAT -- Capabilities

A platform for building conversational voice AI agents with real-time speech interaction, multiple LLM providers, custom tools, and easy web embedding. CLI-first workflow with SDK and REST API alternatives.

**Features:**
- CLI-based agent management (`@elevenlabs/cli`)
- Six starter templates: default, minimal, voice-only, text-only, customer-service, assistant
- Multi-LLM support: OpenAI (GPT-4o), Anthropic (Claude 3.5), Google (Gemini 1.5), custom endpoints
- Three tool types: webhook (server-side), client (browser-side), system (built-in)
- System tools: end_call, transfer_to_number, transfer_to_agent
- Widget embedding with web component (`<elevenlabs-convai>`)
- React hooks (`useConversation`) and browser client (`Conversation.startSession`)
- Outbound phone calls via Twilio integration
- Knowledge base / RAG support
- Platform settings: auth (signed URLs, allowlists), privacy (recording, retention), call limits
- Turn-taking modes: server_vad (auto) and turn_based (manual)

**File inventory:**
- `agents/SKILL.md` -- 266 lines
- `references/installation.md` -- 131 lines
- `references/agent-configuration.md` -- 401 lines
- `references/client-tools.md` -- 435 lines
- `references/widget-embedding.md` -- 365 lines
- `references/outbound-calls.md` -- 153 lines

### 6.2 WHY -- Problem Space

**Problems it solves:**
- Building voice assistants, customer service bots, interactive characters
- Creating real-time voice conversation experiences
- Integrating voice AI into websites with minimal code
- Making outbound phone calls with AI agents

**When to use it:**
- "Build a voice agent"
- "Create a customer service bot"
- "Add voice AI to my website"
- "Make an outbound call"

**When NOT to use it:**
- You just need text-to-speech without conversation (use `text-to-speech`)
- You just need transcription (use `speech-to-text`)

### 6.3 HOW -- Usage

**Minimal example (CLI):**

```bash
npm install -g @elevenlabs/cli
elevenlabs auth login
elevenlabs agents init
elevenlabs agents add "My Assistant" --template default
elevenlabs agents push
```

**Key configuration options:**
- `conversation_config.agent` -- first message, language, max tokens
- `conversation_config.tts` -- voice ID, model, stability, similarity
- `conversation_config.asr` -- speech recognition model, keyterms
- `conversation_config.turn` -- mode (server_vad/turn_based), silence threshold, interrupt sensitivity
- `prompt` -- system prompt, LLM choice, temperature, max tokens
- `tools` -- array of webhook, client, and system tools
- `platform_settings` -- auth, privacy, call limits

**Advanced patterns:**
- Widget embedding: `<elevenlabs-convai agent-id="..."></elevenlabs-convai>` with customizable avatar, colors, text labels, and CSS
- Client tools that execute JavaScript in the browser and return data to the agent
- Outbound calls with per-call configuration overrides and dynamic variables
- Knowledge base integration for RAG-powered agents
- CI/CD deployment with `elevenlabs agents push`

**Error handling:**
- 401: Invalid API key
- 404: Agent not found
- 422: Invalid configuration
- 429: Rate limit exceeded

### 6.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `agents` -- kebab-case, 6 chars, matches directory |
| `description` field | PASS | 141 chars, WHAT + WHEN with trigger phrases |
| Progressive disclosure | PASS | 266-line SKILL.md with 5 deep-dive references |
| Content quality | PASS | CLI + SDK + cURL examples, tools, widget, outbound calls, error handling |
| Security | PASS | Clean YAML |
| Reference structure | PASS | Five focused references covering distinct concerns |

---

## 7. openclaw/elevenlabs-transcribe

### 7.1 WHAT -- Capabilities

A CLI tool for speech-to-text transcription. Unlike the SDK-based `speech-to-text` skill, this is a standalone Python script with a bash wrapper that manages its own virtual environment.

**Features:**
- Batch transcription of local audio/video files
- Real-time streaming from URLs (e.g., live radio, podcasts)
- Real-time transcription from microphone input
- Real-time streaming of local files (for testing)
- Speaker diarization
- Language hints
- Audio event tagging
- JSON output with timestamps
- Partial transcript display
- Quiet mode for agent integration
- Auto-creates and manages a Python virtual environment
- Checks for `ffmpeg` and `python3` dependencies

**Supported formats:** Same as `speech-to-text` (MP3, WAV, M4A, FLAC, OGG, WebM, AAC, AIFF, Opus, MP4, AVI, MKV, MOV, WMV, FLV, MPEG, 3GPP -- up to 3 GB, 10 hours)

**File inventory:**
- `openclaw/elevenlabs-transcribe/SKILL.md` -- 147 lines
- `scripts/transcribe.py` -- 417 lines
- `scripts/transcribe.sh` -- 101 lines
- `scripts/requirements.txt` -- 5 lines (elevenlabs 2.34.0, pydub, python-dotenv, sounddevice, numpy)

### 7.2 WHY -- Problem Space

**Problems it solves:**
- Quick transcription from the command line without writing code
- Live transcription of streaming audio sources
- Voice input for AI agents in CLI environments

**When to use it:**
- You want a ready-made CLI tool, not SDK integration
- You need to transcribe from a microphone or live URL
- You prefer shell scripts over Python/JS code

**When NOT to use it:**
- You are building an application and need SDK integration (use `speech-to-text`)
- You need client-side browser transcription (use `speech-to-text` with React hooks)

### 7.3 HOW -- Usage

**Minimal example:**

```bash
# Batch transcribe a file
./scripts/transcribe.sh recording.mp3

# With speaker identification
./scripts/transcribe.sh meeting.mp3 --diarize

# Real-time from microphone
./scripts/transcribe.sh --mic

# Real-time from URL
./scripts/transcribe.sh --url https://example.com/live.mp3
```

**Key options:**
- `--diarize` -- identify speakers
- `--lang CODE` -- ISO language hint
- `--json` -- full JSON output with timestamps
- `--events` -- tag audio events (laughter, music, applause)
- `--realtime` -- stream local file instead of batch processing
- `--partials` -- show interim transcripts during real-time mode
- `-q, --quiet` -- suppress status messages (recommended for agents)

**Error handling:**
- Missing API key: prints error and exits with non-zero status
- File not found: prints error and exits
- Missing ffmpeg: prints installation instructions and exits
- API errors: propagated from the ElevenLabs SDK

### 7.4 BEST PRACTICES AUDIT

| Criterion | Rating | Notes |
|-----------|--------|-------|
| `name` field | PASS | `elevenlabs-transcribe` -- kebab-case, 22 chars, matches directory |
| `description` field | PASS | 134 chars, WHAT + WHEN |
| Progressive disclosure | PASS | 147-line SKILL.md, implementation in scripts |
| Content quality | PASS | Clear examples for every mode, options table, format list |
| Security | PASS | No XML in YAML |
| Reference structure | PASS | No references needed; scripts serve as implementation |
| **Spec compliance** | PARTIAL | Metadata uses `clawdbot` instead of `openclaw` (see below) |

**Specific findings:**
1. **Inconsistent metadata key**: Uses `"clawdbot"` in the metadata field where the directory structure suggests `"openclaw"`. This is a naming inconsistency.
2. **Missing `license` field**: Unlike all 6 official skills, this skill omits the `license` field from frontmatter.
3. **Missing `compatibility` field**: Also omits the `compatibility` field present in official skills.

---

# Part 2: Cross-Cutting Analysis

---

## Common Patterns

### Frontmatter Structure

All 6 official skills share identical frontmatter structure:

```yaml
name: <kebab-case-name>
description: <WHAT + WHEN trigger phrases>
license: MIT
compatibility: Requires internet access and an ElevenLabs API key (ELEVENLABS_API_KEY).
metadata: {"openclaw": {"requires": {"env": ["ELEVENLABS_API_KEY"]}, "primaryEnv": "ELEVENLABS_API_KEY"}}
```

Exceptions:
- `setup-api-key` has no `metadata` field and a different `compatibility` string
- `openclaw/elevenlabs-transcribe` uses `clawdbot` instead of `openclaw` in metadata, omits `license` and `compatibility`

### SDK Trinity Pattern

Every official skill provides examples in three languages/formats:
1. **Python** -- using `from elevenlabs.client import ElevenLabs`
2. **JavaScript** -- using `import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js"`
3. **cURL** -- using `xi-api-key` header against `https://api.elevenlabs.io/v1/`

### Client Initialization Pattern

All SDK examples follow the same initialization:

```python
# Python
from elevenlabs.client import ElevenLabs
client = ElevenLabs()  # reads ELEVENLABS_API_KEY from environment
```

```javascript
// JavaScript
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
const client = new ElevenLabsClient();  // reads ELEVENLABS_API_KEY from environment
```

### Error Handling Pattern

All skills document the same three HTTP error codes:
- **401**: Invalid API key
- **422**: Invalid parameters
- **429**: Rate limit exceeded

Some skills add context-specific codes (404 for agents, WebSocket error codes for real-time streaming, `bad_prompt`/`bad_composition_plan` for music).

### Installation Reference Duplication

Five skills include a `references/installation.md` file:
- `text-to-speech/references/installation.md` (90 lines)
- `speech-to-text/references/installation.md` (92 lines)
- `sound-effects/references/installation.md` (63 lines)
- `music/references/installation.md` (65 lines)
- `agents/references/installation.md` (131 lines)

The first four are near-identical (JS SDK install, deprecation warning, Python install, cURL setup, API key instructions). The agents version adds CLI installation. This is significant duplication -- any update to SDK instructions must be replicated across 5 files.

---

## Skill Interaction Map

```
                    setup-api-key
                         |
            provides ELEVENLABS_API_KEY to all
                         |
        +--------+-------+-------+--------+
        |        |       |       |        |
   text-to-  speech-  sound-   music   agents
   speech    to-text  effects           |
        |        |                      |
        +--<>----+          internally composes
     bidirectional          TTS + STT + tools
       pipeline
                     |
              speech-to-text (SDK)
                 vs.
              elevenlabs-transcribe (CLI)
                same API, different interface
```

**Key relationships:**
- `setup-api-key` is a prerequisite for all other skills
- `text-to-speech` and `speech-to-text` form a bidirectional pipeline (text-to-audio-to-text)
- `agents` internally composes TTS + STT + LLM + tools into a conversational system
- `speech-to-text` (SDK) vs. `openclaw/elevenlabs-transcribe` (CLI) target different users: SDK integration vs. command-line usage
- `sound-effects` vs. `music` distinguish non-speech audio by structure: unstructured audio textures vs. structured musical compositions

---

## Recommendations for Improvement

### 1. BUG: Unclosed Code Block in speech-to-text/SKILL.md

**File:** `speech-to-text/SKILL.md`, line 157
**Severity:** Medium
**Issue:** The `## Tracking Costs` section opens a Python code block at line 157 that is never closed before the `## Real-Time Streaming` heading. The closing ` ``` ` is missing.

**Fix:** Add a closing ` ``` ` before the `## Real-Time Streaming` heading.

### 2. Inconsistent Metadata Across Skills

**Severity:** Low
**Issue:**
- `setup-api-key` lacks the `metadata` field entirely
- `openclaw/elevenlabs-transcribe` uses `"clawdbot"` instead of `"openclaw"` as the metadata key

**Fix:** Add a `metadata` field to `setup-api-key` and change `clawdbot` to `openclaw` in `elevenlabs-transcribe`.

### 3. Installation Reference Duplication

**Severity:** Low (maintenance burden)
**Issue:** Five near-identical `installation.md` files exist across skills. Any change to SDK installation instructions requires updating 5 files.

**Fix:** Consider either: (a) a shared installation reference that skills link to, or (b) accepting the duplication as a deliberate trade-off for skill self-containment.

### 4. Missing Music Output Formats

**Severity:** Medium
**Issue:** `sound-effects` documents 21 output formats. `music` documents zero. Users have no way to know what formats `music.compose` supports or how to request them.

**Fix:** Add an output formats table to `music/SKILL.md` or `music/references/api_reference.md`.

### 5. Missing Music Prompt Tips

**Severity:** Low
**Issue:** `sound-effects` includes a "Prompt Tips" section with guidance on writing effective prompts. `music` has none.

**Fix:** Add a prompt tips section to `music/SKILL.md` with guidance on describing genre, mood, instrumentation, tempo, etc.

### 6. setup-api-key Security Gaps

**Severity:** Medium
**Issue:**
- No guidance on adding `.env` to `.gitignore` -- users may commit their API key
- No handling of pre-existing `.env` files (could overwrite other variables)

**Fix:** Add a step that checks for `.gitignore` and warns/creates it, and append to `.env` rather than overwriting.

### 7. Stale Model IDs

**Severity:** Low (future risk)
**Issue:** Hard-coded model IDs like `eleven_multilingual_v2`, `scribe_v2`, `music_v1` appear throughout. If ElevenLabs deprecates or renames models, all skills need updating.

**Fix:** Consider noting model ID currency dates or linking to a canonical model list. This is informational -- no immediate action required.

---

# Part 3: Appendix

---

## Complete File Inventory

| File | Lines |
|------|-------|
| `CLAUDE.md` | 52 |
| `README.md` | 48 |
| `setup-api-key/SKILL.md` | 56 |
| `text-to-speech/SKILL.md` | 218 |
| `text-to-speech/references/installation.md` | 90 |
| `text-to-speech/references/streaming.md` | 307 |
| `text-to-speech/references/voice-settings.md` | 115 |
| `speech-to-text/SKILL.md` | 259 |
| `speech-to-text/references/installation.md` | 92 |
| `speech-to-text/references/transcription-options.md` | 174 |
| `speech-to-text/references/realtime-client-side.md` | 169 |
| `speech-to-text/references/realtime-server-side.md` | 316 |
| `speech-to-text/references/realtime-commit-strategies.md` | 124 |
| `speech-to-text/references/realtime-events.md` | 195 |
| `sound-effects/SKILL.md` | 120 |
| `sound-effects/references/installation.md` | 63 |
| `music/SKILL.md` | 105 |
| `music/references/installation.md` | 65 |
| `music/references/api_reference.md` | 165 |
| `agents/SKILL.md` | 266 |
| `agents/references/installation.md` | 131 |
| `agents/references/agent-configuration.md` | 401 |
| `agents/references/client-tools.md` | 435 |
| `agents/references/widget-embedding.md` | 365 |
| `agents/references/outbound-calls.md` | 153 |
| `openclaw/elevenlabs-transcribe/SKILL.md` | 147 |
| `openclaw/elevenlabs-transcribe/scripts/transcribe.py` | 417 |
| `openclaw/elevenlabs-transcribe/scripts/transcribe.sh` | 101 |
| `openclaw/elevenlabs-transcribe/scripts/requirements.txt` | 5 |

**Total:** 29 files, ~4,958 lines

## Compliance Summary Matrix

| Skill | name | description | Progressive Disclosure | Content Quality | Security | References |
|-------|------|-------------|----------------------|-----------------|----------|------------|
| setup-api-key | PASS | PASS | PASS | PARTIAL | PASS | PASS |
| text-to-speech | PASS | PASS | PASS | PASS | PASS | PASS |
| speech-to-text | PASS | PASS | PASS | PARTIAL | PASS | PASS |
| sound-effects | PASS | PASS | PASS | PASS | PASS | PASS |
| music | PASS | PASS | PASS | PARTIAL | PASS | PASS |
| agents | PASS | PASS | PASS | PASS | PASS | PASS |
| elevenlabs-transcribe | PASS | PASS | PASS | PASS | PASS | PASS |

**Summary:** 7/7 skills pass on name, description, progressive disclosure, security, and references. 3/7 have partial content quality ratings due to the unclosed code block (speech-to-text), missing output formats/prompt tips (music), and missing `.gitignore` guidance (setup-api-key).

## Key Terms Glossary

| Term | Definition |
|------|-----------|
| **Agent Skill** | A self-contained capability package following the agentskills.io specification |
| **SKILL.md** | The main skill definition file with YAML frontmatter and quick-start content |
| **Frontmatter** | YAML metadata block at the top of SKILL.md (name, description, license, etc.) |
| **Progressive disclosure** | Keeping SKILL.md concise and delegating details to reference files |
| **Scribe v2** | ElevenLabs' speech-to-text model for batch transcription |
| **VAD** | Voice Activity Detection -- auto-commits transcripts when silence is detected |
| **Composition plan** | A structured description of music sections, styles, and lyrics for granular control |
| **Request stitching** | Providing context between sequential TTS requests for seamless audio boundaries |
| **Diarization** | Identifying and labeling different speakers in an audio recording |
| **Keyterm prompting** | Providing domain-specific terms to improve transcription accuracy |
