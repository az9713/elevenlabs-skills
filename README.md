> **This is a fork/clone of [`elevenlabs/skills`](https://github.com/elevenlabs/skills) — the official ElevenLabs Agent Skills repository.**
> All original skills are preserved. This fork adds a comprehensive **[Skills Guide](./SKILLS_GUIDE.md)** (what/why/how + best-practices audit for every skill), **[Workflow Ideas](./WORKFLOW_IDEAS.md)** (14 foundational multi-skill workflows), and **[Advanced Workflows](./ADVANCED_WORKFLOWS.md)** (10 production-grade workflows leveraging 2026 platform capabilities: C2PA signing, MCP tools, guardrails, agent version control, A/B testing, and more).

## Playground — Working Implementations

The [`playground/`](./playground) directory contains fully tested, runnable implementations of select workflows. These go beyond documentation — they are real Python scripts you can run today with an ElevenLabs API key.

| Project | Source Workflow | What It Does |
|---------|----------------|--------------|
| [**Audio Adventure**](./playground/audio-adventure) | [Workflow 3](./WORKFLOW_IDEAS.md) | A voice-controlled text adventure where you speak into your mic, an AI Dungeon Master narrates back in real-time, and background music, ambient sounds, and sound effects are generated dynamically. Built on the Conversational AI SDK with pygame for layered audio mixing. Includes a [soundtrack recording](./playground/audio-adventure/audio_adventure_soundtrack.m4a) from a test session. |
| [**Voice Migration**](./playground/voice-migration) | [Workflow 19](./ADVANCED_WORKFLOWS.md) | A pipeline that swaps a speaker's voice in multi-speaker audio while preserving the original delivery, timing, and conversation structure. Chains Speech-to-Text (with diarization), Speech-to-Speech, and audio reassembly — tested at under 2% timing drift. |

---

![LOGO](/logo.png)

# ElevenLabs Skills

Agent skills for [ElevenLabs](https://elevenlabs.io) developer products. These skills follow the [Agent Skills specification](https://agentskills.io/specification) and can be used with any compatible AI coding assistant.

## Installation

```bash
npx skills add elevenlabs/skills
```

## Available Skills

| Skill | Description |
|-------|-------------|
| [text-to-speech](./text-to-speech) | Convert text to lifelike speech using ElevenLabs' AI voices |
| [speech-to-text](./speech-to-text) | Transcribe audio files to text with timestamps |
| [agents](./agents) | Build conversational voice AI agents |
| [sound-effects](./sound-effects) | Generate sound effects from text descriptions |
| [music](./music) | Generate music tracks using AI composition |
| [setup-api-key](./setup-api-key) | Guide through obtaining and configuring an ElevenLabs API key |

## Configuration

All skills require an ElevenLabs API key. Set it as an environment variable:

```bash
export ELEVENLABS_API_KEY="your-api-key"
```

Get your API key from the `setup-api-key` skill or use the [ElevenLabs dashboard](https://elevenlabs.io/app/settings/api-keys).

## SDK Support

Most skills include examples for:

- **Python** - `pip install elevenlabs`
- **JavaScript/TypeScript** - `npm install @elevenlabs/elevenlabs-js`
- **cURL** - Direct REST API calls

> **JavaScript SDK Warning:** Always use `@elevenlabs/elevenlabs-js`. Do not use `npm install elevenlabs` (that's an outdated v1.x package).

See the installation guide in any skill's `references/` folder for complete setup instructions including migration from deprecated packages.

## License

MIT
