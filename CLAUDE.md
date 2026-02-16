# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **ElevenLabs Skills** repository — a collection of AI agent skills following the [Agent Skills specification](https://agentskills.io/specification). It is not a traditional software project with build systems or test suites. It is a documentation-and-configuration-driven project distributed via `npx skills add elevenlabs/skills`.

## Repository Structure

Each skill is a self-contained directory with:
- `SKILL.md` — Main skill definition with YAML frontmatter (name, description, license, compatibility, metadata) and quick start examples in Python, JavaScript, and cURL
- `references/` — Detailed reference documentation (installation, streaming, configuration, etc.)
- `scripts/` — Executable implementations (only present in `openclaw/elevenlabs-transcribe/`)

**Skills:**
| Directory | Purpose |
|-----------|---------|
| `text-to-speech/` | Convert text to speech (74+ languages, multiple models) |
| `speech-to-text/` | Transcribe audio to text (90+ languages, batch and real-time) |
| `agents/` | Build conversational voice AI agents (CLI-first, multi-LLM) |
| `sound-effects/` | Generate sound effects from text descriptions |
| `music/` | Generate music tracks via AI composition |
| `setup-api-key/` | Guide for obtaining and configuring an ElevenLabs API key |
| `openclaw/elevenlabs-transcribe/` | CLI transcription script (Python) |

## Key Conventions

- **SKILL.md frontmatter** must include `name`, `description`, `license`, `compatibility`, and `metadata` fields in YAML
- All skills require `ELEVENLABS_API_KEY` as an environment variable
- API base URL: `https://api.elevenlabs.io/v1/` with header `xi-api-key: $ELEVENLABS_API_KEY`
- SDK examples always show Python (`elevenlabs`), JavaScript (`@elevenlabs/elevenlabs-js`), and cURL
- **JavaScript SDK warning:** Always use `@elevenlabs/elevenlabs-js`. Never use `npm install elevenlabs` (outdated v1.x package)

## Only Executable Code

The only runnable code is `openclaw/elevenlabs-transcribe/scripts/transcribe.py` (Python):
- Dependencies in `openclaw/elevenlabs-transcribe/scripts/requirements.txt`: `elevenlabs==2.34.0`, `pydub==0.25.1`, `python-dotenv==1.0.1`, `sounddevice==0.5.1`, `numpy>=1.24.0`
- Requires `ffmpeg` and `python3` binaries
- Supports batch transcription, real-time URL streaming, and microphone input

## Agents CLI

The agents skill uses the ElevenLabs CLI as its primary interface:
```bash
npm install -g @elevenlabs/cli
elevenlabs auth login
elevenlabs agents init
elevenlabs agents add "Name" --template default
elevenlabs agents push
```
Templates: `default`, `minimal`, `voice-only`, `text-only`, `customer-service`, `assistant`
