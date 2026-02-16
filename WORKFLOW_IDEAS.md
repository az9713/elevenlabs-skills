# Creative Multi-Skill Workflows

Fourteen production-ready workflows that combine ElevenLabs skills and APIs in unexpected ways. Each workflow lists the skills it chains together, explains the motivation, and provides enough implementation detail to build it.

**Skills referenced:** setup-api-key, text-to-speech, speech-to-text, sound-effects, music, agents, elevenlabs-transcribe

**APIs beyond skills:** Text-to-Dialogue (`/v1/text-to-dialogue`), Voice Design (`/v1/text-to-voice/design`), Dubbing (`/v1/dubbing`), Audio Isolation (`/v1/audio-isolation`), Agent WebSocket monitoring (`/v1/convai/conversations/{id}/monitor`), Pronunciation Dictionaries

---

## 1. The AI Podcast Factory

**Skills:** `text-to-speech` + `music` + `sound-effects` + `speech-to-text`

### Motivation

Producing a podcast normally requires recording, editing, mixing music, adding sound effects, and writing show notes. This workflow generates an entire podcast episode from a topic and a bullet-point outline -- two AI hosts debate the topic with distinct voices, backed by a generated theme song and transition stingers, with a full searchable transcript at the end.

### Architecture

```
Topic + Outline
      |
      v
 LLM (GPT-4o / Claude)  -->  script with [HOST_A], [HOST_B] tags,
                               [MUSIC:description], [SFX:description] cues
      |
      v
 +----+----+----+
 |         |    |
 v         v    v
TTS x2   Music  SFX
(two      (intro (transition
voices)   theme) stingers)
 |         |    |
 v         v    v
   Audio Mixer (pydub)
      |
      v
  Final MP3 + Transcript (via STT)
```

### Implementation

```python
import json
import re
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import io

client = ElevenLabs()

# --- Step 1: Generate the script with an LLM ---
# (Use your preferred LLM -- this is the prompt structure)
SCRIPT_PROMPT = """
Write a 3-minute podcast script about {topic}.
Two hosts: Alex (curious, asks questions) and Morgan (expert, explains).
Format each line as:
  [HOST_A] dialogue text
  [HOST_B] dialogue text
  [MUSIC:description] for music cues
  [SFX:description] for sound effect cues
Start with [MUSIC:upbeat podcast intro theme, 8 seconds].
Add [SFX:whoosh transition] between segments.
End with [MUSIC:mellow outro, 5 seconds].
"""

# Assume `script` is the LLM output -- parse it:
def parse_script(script: str) -> list[dict]:
    """Parse script into typed segments."""
    segments = []
    for line in script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if m := re.match(r"\[HOST_A\]\s*(.*)", line):
            segments.append({"type": "speech", "voice": "host_a", "text": m.group(1)})
        elif m := re.match(r"\[HOST_B\]\s*(.*)", line):
            segments.append({"type": "speech", "voice": "host_b", "text": m.group(1)})
        elif m := re.match(r"\[MUSIC:(.*?)\]", line):
            segments.append({"type": "music", "prompt": m.group(1)})
        elif m := re.match(r"\[SFX:(.*?)\]", line):
            segments.append({"type": "sfx", "prompt": m.group(1)})
    return segments

# --- Step 2: Generate audio for each segment ---
VOICES = {
    "host_a": {"id": "JBFqnCBsd6RMkjVDRZzb", "settings": VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.3)},
    "host_b": {"id": "EXAVITQu4vr4xnSDxMaL", "settings": VoiceSettings(stability=0.5, similarity_boost=0.7, style=0.2)},
}

def generate_speech(text: str, voice_key: str) -> AudioSegment:
    voice = VOICES[voice_key]
    audio_iter = client.text_to_speech.convert(
        text=text,
        voice_id=voice["id"],
        model_id="eleven_multilingual_v2",
        voice_settings=voice["settings"],
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_iter)
    return AudioSegment.from_mp3(io.BytesIO(audio_bytes))

def generate_music(prompt: str) -> AudioSegment:
    # Extract duration hint from prompt if present (e.g., "8 seconds")
    duration_ms = 10000  # default 10s
    if m := re.search(r"(\d+)\s*seconds?", prompt):
        duration_ms = int(m.group(1)) * 1000
    audio_iter = client.music.compose(prompt=prompt, music_length_ms=duration_ms)
    audio_bytes = b"".join(audio_iter)
    return AudioSegment.from_mp3(io.BytesIO(audio_bytes))

def generate_sfx(prompt: str) -> AudioSegment:
    audio_iter = client.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=2.0,
        prompt_influence=0.6,
    )
    audio_bytes = b"".join(audio_iter)
    return AudioSegment.from_mp3(io.BytesIO(audio_bytes))

# --- Step 3: Assemble the episode ---
def assemble_podcast(segments: list[dict]) -> AudioSegment:
    episode = AudioSegment.silent(duration=0)
    for seg in segments:
        if seg["type"] == "speech":
            clip = generate_speech(seg["text"], seg["voice"])
            episode += clip + AudioSegment.silent(duration=300)  # 300ms pause
        elif seg["type"] == "music":
            clip = generate_music(seg["prompt"]) - 8  # reduce volume by 8dB
            episode += clip
        elif seg["type"] == "sfx":
            clip = generate_sfx(seg["prompt"])
            episode += clip
    return episode

# --- Step 4: Export and transcribe ---
def produce_episode(script: str, output_path: str):
    segments = parse_script(script)
    episode = assemble_podcast(segments)
    episode.export(output_path, format="mp3")

    # Generate searchable transcript
    with open(output_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f, model_id="scribe_v2", diarize=True, timestamps_granularity="word"
        )
    transcript_path = output_path.replace(".mp3", "_transcript.json")
    with open(transcript_path, "w") as f:
        json.dump({"text": result.text, "words": [
            {"text": w.text, "start": w.start, "end": w.end, "speaker": w.speaker_id}
            for w in result.words if w.type == "word"
        ]}, f, indent=2)
    return output_path, transcript_path
```

### What makes this creative

A single text prompt produces a complete, ready-to-publish podcast episode with two distinct AI voices, original theme music, transition effects, and a searchable transcript -- all without a microphone, DAW, or music library.

---

## 2. Live Translation Booth

**Skills:** `speech-to-text` (real-time) + `text-to-speech` (streaming)

### Motivation

Real-time translation normally requires human interpreters or expensive hardware. This workflow creates a software-only simultaneous translation booth: speak in one language, hear the translation in another language with near-real-time latency. Useful for multilingual meetings, live events, or language learning.

### Architecture

```
Microphone (source language)
      |
      v
  STT real-time (scribe_v2_realtime)
      |  committed transcripts
      v
  Translation LLM (streaming)
      |  translated text chunks
      v
  TTS WebSocket (eleven_flash_v2_5)
      |  audio chunks
      v
  Speaker (target language)
```

### Implementation

```python
import asyncio
import json
import base64
import os
import websockets
import sounddevice as sd
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs import AudioFormat, CommitStrategy, RealtimeAudioOptions, RealtimeEvents
from openai import OpenAI  # or any LLM with streaming

client = ElevenLabs()
openai_client = OpenAI()

SAMPLE_RATE = 16000
TARGET_LANG = "Spanish"
TTS_VOICE = "JBFqnCBsd6RMkjVDRZzb"

async def translation_booth():
    """Run the full translation pipeline."""
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    text_queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # --- STT: Microphone -> Text ---
    connection = await client.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=SAMPLE_RATE,
            commit_strategy=CommitStrategy.VAD,
        )
    )

    def mic_callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy().tobytes())

    async def send_mic_audio():
        while True:
            data = await audio_queue.get()
            chunk_b64 = base64.b64encode(data).decode()
            await connection.send({"audio_base_64": chunk_b64, "sample_rate": SAMPLE_RATE})

    def on_committed(data):
        text = data.get("text", "").strip()
        if text:
            loop.call_soon_threadsafe(text_queue.put_nowait, text)

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
    connection.on(RealtimeEvents.SESSION_STARTED, lambda _: asyncio.create_task(send_mic_audio()))

    # --- Translation: Source Text -> Target Text ---
    async def translate_and_speak():
        ws_uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE}"
            f"/stream-input?model_id=eleven_flash_v2_5"
        )
        async with websockets.connect(ws_uri) as tts_ws:
            # Initialize TTS WebSocket
            await tts_ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                "xi_api_key": os.getenv("ELEVENLABS_API_KEY"),
            }))

            # Start a task to play received audio
            async def play_audio():
                while True:
                    msg = await tts_ws.recv()
                    data = json.loads(msg)
                    if data.get("audio"):
                        pcm = base64.b64decode(data["audio"])
                        # Play audio through speakers (simplified)
                        print("[SPEAKING]", end="", flush=True)
                    elif data.get("isFinal"):
                        break

            play_task = asyncio.create_task(play_audio())

            while True:
                source_text = await text_queue.get()
                print(f"\n[HEARD] {source_text}")

                # Stream translation from LLM
                stream = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"Translate to {TARGET_LANG}. Output ONLY the translation."},
                        {"role": "user", "content": source_text},
                    ],
                    stream=True,
                )

                translated = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    translated += delta
                    if delta:
                        # Stream each text chunk to TTS WebSocket
                        await tts_ws.send(json.dumps({"text": delta}))

                # Flush the TTS buffer at sentence boundary
                await tts_ws.send(json.dumps({"text": " ", "flush": True}))
                print(f"[TRANSLATED] {translated}")

    # --- Run everything ---
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=int(SAMPLE_RATE * 0.1), callback=mic_callback):
        print("Translation booth active. Speak into your microphone...")
        await asyncio.gather(translate_and_speak())

# asyncio.run(translation_booth())
```

### Key design decisions

- **VAD commit strategy** on STT so translations trigger at natural speech pauses, not mid-sentence
- **WebSocket TTS** (not REST) so translated text streams directly to audio as the LLM generates it -- minimizing total latency
- **Flash model** (`eleven_flash_v2_5`) for ~75ms TTS latency
- Total pipeline latency: ~150ms (STT) + ~200ms (LLM first token) + ~75ms (TTS) = **~425ms** end-to-end

### What makes this creative

Chains the fastest variants of STT and TTS with a streaming LLM to create a real-time simultaneous interpreter. The WebSocket TTS connection stays open across multiple utterances, avoiding connection overhead.

---

## 3. Immersive Audio Adventure Engine

**Skills:** `agents` + `sound-effects` + `music` + `text-to-speech`

### Motivation

Text-based interactive fiction is a proven genre, but voice-based interactive fiction with dynamic soundscapes barely exists. This workflow creates an agent that narrates a branching story, generates contextual background music and sound effects in real time, and responds to the player's spoken choices.

### Architecture

```
Player speaks choice
      |
      v
  Agent (voice AI with custom tools)
      |
      +---> tool: set_scene(mood, environment)
      |         |
      |         +---> Music: generate background track
      |         +---> SFX: generate ambient sounds
      |
      +---> tool: play_effect(description)
      |         |
      |         +---> SFX: one-shot effect
      |
      +---> Agent narrates next scene (TTS built into agent)
      |
      v
  Player hears: narration + music + effects layered together
```

### Implementation

**Step 1: Create the agent with custom tools**

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()

agent = client.conversational_ai.agents.create(
    name="Dungeon Master",
    conversation_config={
        "agent": {
            "first_message": (
                "Welcome, adventurer. You stand at the entrance of a crumbling castle. "
                "Torchlight flickers through the iron gate. What do you do?"
            ),
            "language": "en",
        },
        "tts": {
            "voice_id": "onwK4e9ZLuTAKqWW03F9",  # Daniel - authoritative
            "model_id": "eleven_flash_v2_5",
        },
        "turn": {
            "mode": "server_vad",
            "silence_threshold_ms": 1500,  # longer pause for thoughtful responses
        },
    },
    prompt={
        "prompt": """You are an immersive dungeon master narrating an interactive audio adventure.

Rules:
- Describe scenes vividly using sound-oriented language (what the player HEARS)
- After each scene description, call set_scene with the mood and environment
- When dramatic events happen (door creaking, sword clashing), call play_effect
- Present 2-3 choices after each scene
- Track the player's inventory and health mentally
- Keep responses under 4 sentences for pacing

IMPORTANT: Call set_scene at the START of each new location. Call play_effect for
dramatic moments DURING narration.""",
        "llm": "gpt-4o",
        "temperature": 0.8,
    },
    tools=[
        {
            "type": "webhook",
            "name": "set_scene",
            "description": (
                "Set the background atmosphere for the current scene. "
                "Call this when the player enters a new area or the mood shifts."
            ),
            "webhook": {
                "url": "https://your-server.com/api/adventure/set-scene",
                "method": "POST",
                "timeout_ms": 15000,  # music generation takes time
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "Emotional tone: tense, mysterious, triumphant, peaceful, terrifying",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Physical setting: castle_dungeon, dark_forest, throne_room, cave, battlefield",
                    },
                },
                "required": ["mood", "environment"],
            },
        },
        {
            "type": "webhook",
            "name": "play_effect",
            "description": (
                "Play a one-shot sound effect for a dramatic moment. "
                "Call this for events like doors opening, swords clashing, explosions."
            ),
            "webhook": {
                "url": "https://your-server.com/api/adventure/play-effect",
                "method": "POST",
                "timeout_ms": 8000,
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Vivid description of the sound effect to generate",
                    },
                },
                "required": ["description"],
            },
        },
    ],
)

print(f"Agent created: {agent.agent_id}")
```

**Step 2: Webhook server that generates audio**

```python
from fastapi import FastAPI, Request
from elevenlabs.client import ElevenLabs
import io

app = FastAPI()
client = ElevenLabs()

# Cache generated music to avoid regenerating for the same scene
music_cache: dict[str, bytes] = {}

SCENE_PROMPTS = {
    ("tense", "castle_dungeon"): "Dark ambient drone with distant water dripping, ominous low strings, 15 seconds",
    ("mysterious", "dark_forest"): "Ethereal forest ambience with owl hoots and gentle wind, mystical pads, 15 seconds",
    ("triumphant", "throne_room"): "Heroic orchestral fanfare with brass and timpani, 10 seconds",
    ("terrifying", "cave"): "Horror ambience with echoing whispers and deep rumbling, 15 seconds",
    ("peaceful", "dark_forest"): "Gentle woodland ambience with birdsong and soft harp, 15 seconds",
}

@app.post("/api/adventure/set-scene")
async def set_scene(request: Request):
    data = await request.json()
    mood = data["parameters"]["mood"]
    env = data["parameters"]["environment"]

    cache_key = f"{mood}_{env}"
    if cache_key not in music_cache:
        prompt = SCENE_PROMPTS.get(
            (mood, env),
            f"{mood} atmospheric background music for a {env.replace('_', ' ')}, 15 seconds"
        )

        # Generate background music
        music_audio = b"".join(client.music.compose(
            prompt=prompt, music_length_ms=15000
        ))

        # Generate ambient sound effect
        sfx_audio = b"".join(client.text_to_sound_effects.convert(
            text=f"Ambient sounds of a {env.replace('_', ' ')}, {mood} atmosphere",
            duration_seconds=10.0,
            loop=True,
            prompt_influence=0.5,
        ))

        music_cache[cache_key] = music_audio
        # In production: mix and send to the client's audio player

    return {"result": f"Scene set: {mood} {env}. Background audio playing."}

@app.post("/api/adventure/play-effect")
async def play_effect(request: Request):
    data = await request.json()
    description = data["parameters"]["description"]

    audio = b"".join(client.text_to_sound_effects.convert(
        text=description,
        duration_seconds=2.0,
        prompt_influence=0.7,
    ))
    # In production: send audio to the client's audio player via WebSocket

    return {"result": f"Sound effect played: {description}"}
```

### What makes this creative

The agent doesn't just talk -- it orchestrates an entire audio experience. The LLM decides when the mood shifts and what sounds to play, creating a dynamic soundscape that responds to the player's choices. No two playthroughs sound the same.

---

## 4. Voice-Powered Mood Journal

**Skills:** `speech-to-text` + `text-to-speech` + `music` + `sound-effects`

### Motivation

Journaling is proven to improve mental health, but many people find writing tedious. This workflow lets users speak their thoughts, then transforms the entry into a rich audio artifact: a narrated reflection with mood-matched music and ambient sounds. Over time, it builds an audio diary that captures not just words but emotional tone.

### Architecture

```
User speaks freely into microphone
      |
      v
  STT (scribe_v2) --> raw transcript
      |
      v
  LLM Analysis:
    - Extract mood (calm, anxious, joyful, reflective, frustrated)
    - Summarize key themes
    - Write a gentle, reflective narration
      |
      +---> Music: mood-matched ambient track
      +---> SFX: environment matching the mood (rain for melancholy, birds for joy)
      +---> TTS: narrate the reflection in a warm voice
      |
      v
  Audio Mixer --> "Journal Entry: February 16, 2026.mp3"
```

### Implementation

```python
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import io, json
from openai import OpenAI

el_client = ElevenLabs()
llm = OpenAI()

MOOD_MAP = {
    "calm": {
        "music": "Soft ambient piano with gentle pads, peaceful and meditative",
        "sfx": "Gentle rain on a window with distant soft thunder",
        "voice_settings": VoiceSettings(stability=0.7, similarity_boost=0.6, style=0.1),
    },
    "joyful": {
        "music": "Light acoustic guitar with warm ukulele, uplifting and cheerful",
        "sfx": "Morning birdsong in a sunlit garden with a gentle breeze",
        "voice_settings": VoiceSettings(stability=0.5, similarity_boost=0.7, style=0.3),
    },
    "anxious": {
        "music": "Slow deep breathing rhythm with soft synth pads, calming and grounding",
        "sfx": "Crackling fireplace with occasional wood pops",
        "voice_settings": VoiceSettings(stability=0.6, similarity_boost=0.6, style=0.0),
    },
    "reflective": {
        "music": "Solo cello with minimal piano accompaniment, introspective",
        "sfx": "Ocean waves gently rolling onto a sandy beach",
        "voice_settings": VoiceSettings(stability=0.6, similarity_boost=0.7, style=0.2),
    },
    "frustrated": {
        "music": "Lo-fi downtempo beat with vinyl crackle, mellow and accepting",
        "sfx": "Steady rain with distant city ambience, cozy indoor atmosphere",
        "voice_settings": VoiceSettings(stability=0.5, similarity_boost=0.65, style=0.1),
    },
}

def create_journal_entry(audio_file_path: str, date: str) -> str:
    # 1. Transcribe the spoken journal entry
    with open(audio_file_path, "rb") as f:
        transcript = el_client.speech_to_text.convert(file=f, model_id="scribe_v2")

    # 2. Analyze mood and generate narration
    analysis = llm.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """Analyze this journal entry and return JSON:
{
  "mood": "calm|joyful|anxious|reflective|frustrated",
  "themes": ["theme1", "theme2"],
  "narration": "A gentle 2-3 sentence reflection that acknowledges what was said and offers a compassionate reframing. Write in second person. Do not judge."
}"""},
            {"role": "user", "content": transcript.text},
        ],
    )
    result = json.loads(analysis.choices[0].message.content)
    mood = result["mood"]
    narration = result["narration"]
    config = MOOD_MAP.get(mood, MOOD_MAP["reflective"])

    # 3. Generate all audio in parallel (conceptually -- sequential here for clarity)
    # Intro narration
    intro_text = f"Journal entry. {date}."
    intro_audio = b"".join(el_client.text_to_speech.convert(
        text=intro_text,
        voice_id="XB0fDUnXU5powFXDhCwa",  # Charlotte - warm
        model_id="eleven_multilingual_v2",
        voice_settings=config["voice_settings"],
    ))

    # Reflection narration
    reflection_audio = b"".join(el_client.text_to_speech.convert(
        text=narration,
        voice_id="XB0fDUnXU5powFXDhCwa",
        model_id="eleven_multilingual_v2",
        voice_settings=config["voice_settings"],
    ))

    # Background music (30 seconds)
    music_audio = b"".join(el_client.music.compose(
        prompt=config["music"],
        music_length_ms=30000,
        force_instrumental=True,
    ))

    # Ambient sound (looping, 15 seconds)
    sfx_audio = b"".join(el_client.text_to_sound_effects.convert(
        text=config["sfx"],
        duration_seconds=15.0,
        loop=True,
        prompt_influence=0.5,
    ))

    # 4. Mix everything together
    intro_seg = AudioSegment.from_mp3(io.BytesIO(intro_audio))
    reflection_seg = AudioSegment.from_mp3(io.BytesIO(reflection_audio))
    music_seg = AudioSegment.from_mp3(io.BytesIO(music_audio)) - 14  # quiet background
    sfx_seg = AudioSegment.from_mp3(io.BytesIO(sfx_audio)) - 18     # very quiet ambient

    # Layer: ambient + music as bed, then narration on top
    bed = sfx_seg.overlay(music_seg)
    total_duration = len(intro_seg) + 1000 + len(reflection_seg) + 2000
    # Loop bed to cover total duration
    while len(bed) < total_duration:
        bed += bed
    bed = bed[:total_duration]

    # Overlay speech onto the bed
    final = bed.overlay(intro_seg, position=500)
    final = final.overlay(reflection_seg, position=len(intro_seg) + 1500)

    # Fade in and out
    final = final.fade_in(2000).fade_out(3000)

    output_path = f"journal_{date.replace(' ', '_')}.mp3"
    final.export(output_path, format="mp3")
    return output_path
```

### What makes this creative

Turns a raw voice recording into a produced, emotionally resonant audio artifact. The mood detection drives every creative decision -- music genre, ambient sounds, voice warmth. Over time, you build a library of audio journal entries that sound like a personal podcast.

---

## 5. Meeting Intelligence Pipeline

**Skills:** `elevenlabs-transcribe` (CLI) + `speech-to-text` + `text-to-speech`

### Motivation

Meetings produce hours of audio but minutes of actionable content. This workflow transcribes a meeting with speaker identification, extracts action items and decisions with an LLM, then generates a "5-minute briefing" audio summary that busy stakeholders can listen to during their commute.

### Architecture

```
Meeting recording (MP3/WAV)
      |
      v
  elevenlabs-transcribe --diarize --json
      |
      v
  Diarized transcript JSON
      |
      v
  LLM extracts:
    - Executive summary (3-5 sentences)
    - Key decisions (bulleted)
    - Action items (who, what, when)
    - Unresolved questions
      |
      v
  TTS generates audio briefing:
    - Narrator voice for summary
    - Different voice for action items (emphasis)
      |
      v
  "Meeting Briefing - Feb 16.mp3"
```

### Implementation

```bash
#!/usr/bin/env bash
# Step 1: Transcribe with speaker labels
./openclaw/elevenlabs-transcribe/scripts/transcribe.sh \
  meeting_recording.mp3 --diarize --json > meeting_transcript.json
```

```python
import json
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import io
from openai import OpenAI

el = ElevenLabs()
llm = OpenAI()

# Step 2: Load transcript and analyze
with open("meeting_transcript.json") as f:
    transcript = json.load(f)

# Build speaker-labeled text
speaker_text = []
current_speaker = None
current_words = []
for word in transcript.get("words", []):
    if word.get("type") != "word":
        continue
    speaker = word.get("speaker_id", "unknown")
    if speaker != current_speaker:
        if current_words:
            speaker_text.append(f"[{current_speaker}]: {' '.join(current_words)}")
        current_speaker = speaker
        current_words = [word["text"]]
    else:
        current_words.append(word["text"])
if current_words:
    speaker_text.append(f"[{current_speaker}]: {' '.join(current_words)}")

labeled_transcript = "\n".join(speaker_text)

# Step 3: Extract intelligence with LLM
analysis = llm.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": """Analyze this meeting transcript and return JSON:
{
  "summary": "3-5 sentence executive summary",
  "decisions": ["Decision 1", "Decision 2"],
  "action_items": [
    {"owner": "Speaker 0", "task": "description", "deadline": "if mentioned"}
  ],
  "open_questions": ["Question 1"]
}"""},
        {"role": "user", "content": labeled_transcript},
    ],
)
intel = json.loads(analysis.choices[0].message.content)

# Step 4: Generate audio briefing
NARRATOR = "onwK4e9ZLuTAKqWW03F9"    # Daniel - authoritative
HIGHLIGHT = "EXAVITQu4vr4xnSDxMaL"   # Sarah - clear emphasis

def speak(text: str, voice_id: str) -> AudioSegment:
    audio = b"".join(el.text_to_speech.convert(
        text=text, voice_id=voice_id, model_id="eleven_multilingual_v2",
    ))
    return AudioSegment.from_mp3(io.BytesIO(audio))

# Build the briefing
briefing = AudioSegment.silent(duration=500)

# Summary
briefing += speak("Here's your meeting briefing.", NARRATOR)
briefing += AudioSegment.silent(500)
briefing += speak(intel["summary"], NARRATOR)
briefing += AudioSegment.silent(800)

# Decisions
if intel["decisions"]:
    briefing += speak(f"{len(intel['decisions'])} key decisions were made.", NARRATOR)
    briefing += AudioSegment.silent(300)
    for decision in intel["decisions"]:
        briefing += speak(decision, HIGHLIGHT)
        briefing += AudioSegment.silent(400)

# Action items
if intel["action_items"]:
    briefing += speak(f"{len(intel['action_items'])} action items.", NARRATOR)
    briefing += AudioSegment.silent(300)
    for item in intel["action_items"]:
        text = f"{item['owner']}: {item['task']}"
        if item.get("deadline"):
            text += f", due {item['deadline']}"
        briefing += speak(text, HIGHLIGHT)
        briefing += AudioSegment.silent(400)

# Open questions
if intel.get("open_questions"):
    briefing += speak("Still unresolved:", NARRATOR)
    for q in intel["open_questions"]:
        briefing += speak(q, NARRATOR)
        briefing += AudioSegment.silent(300)

briefing += speak("End of briefing.", NARRATOR)
briefing.export("meeting_briefing.mp3", format="mp3")
```

### What makes this creative

Turns a 60-minute meeting recording into a 3-5 minute audio briefing with structured intelligence. Uses the CLI transcription tool for the heavy lifting and two distinct voices to separate summary narration from action items.

---

## 6. Ambient Workspace Composer

**Skills:** `sound-effects` + `music` + `agents`

### Motivation

Focus music apps play the same loops. This workflow creates an AI agent that generates a personalized, ever-evolving ambient workspace soundscape. Tell it "I need deep focus for coding" or "light background for reading" and it composes original music and layers contextual ambient sounds -- then adapts when you say "I'm getting tired, energize me."

### Implementation

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()

agent = client.conversational_ai.agents.create(
    name="Ambient DJ",
    conversation_config={
        "agent": {
            "first_message": (
                "Hey! I'm your ambient workspace composer. "
                "Tell me what you're working on and how you want to feel, "
                "and I'll create the perfect soundscape."
            ),
            "language": "en",
        },
        "tts": {
            "voice_id": "XB0fDUnXU5powFXDhCwa",  # Charlotte - warm, unobtrusive
            "model_id": "eleven_flash_v2_5",
        },
        "turn": {"mode": "server_vad", "silence_threshold_ms": 2000},
    },
    prompt={
        "prompt": """You are an ambient sound designer. You create focus soundscapes.

When the user describes their activity or mood:
1. Call generate_soundscape with a music prompt and ambient sound description
2. Briefly confirm what you're creating (1 sentence max, keep it calm)
3. When the user wants changes, call generate_soundscape again with updated parameters

Music prompts should describe: genre, tempo (BPM), instruments, mood.
Ambient prompts should describe: environment, specific sounds, intensity.

Keep your voice responses very short and calm -- the user is trying to focus.
Never speak for more than 10 seconds.""",
        "llm": "gpt-4o-mini",
        "temperature": 0.6,
    },
    tools=[
        {
            "type": "webhook",
            "name": "generate_soundscape",
            "description": "Generate a new ambient soundscape with music and environmental sounds",
            "webhook": {
                "url": "https://your-server.com/api/soundscape/generate",
                "method": "POST",
                "timeout_ms": 20000,
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "music_prompt": {
                        "type": "string",
                        "description": "Detailed music description with genre, tempo, instruments, mood",
                    },
                    "ambient_prompt": {
                        "type": "string",
                        "description": "Environmental sound description",
                    },
                    "energy_level": {
                        "type": "string",
                        "enum": ["very_low", "low", "medium", "high"],
                        "description": "Overall energy level of the soundscape",
                    },
                },
                "required": ["music_prompt", "ambient_prompt", "energy_level"],
            },
        },
    ],
)
```

**Webhook server:**

```python
@app.post("/api/soundscape/generate")
async def generate_soundscape(request: Request):
    data = await request.json()
    params = data["parameters"]

    # Duration based on energy level (lower energy = longer, more patient loops)
    durations = {"very_low": 120000, "low": 90000, "medium": 60000, "high": 45000}
    music_ms = durations.get(params["energy_level"], 60000)

    # Generate both in parallel (using asyncio.gather in production)
    music = b"".join(client.music.compose(
        prompt=params["music_prompt"],
        music_length_ms=music_ms,
        force_instrumental=True,
    ))

    ambient = b"".join(client.text_to_sound_effects.convert(
        text=params["ambient_prompt"],
        duration_seconds=30.0,
        loop=True,
        prompt_influence=0.4,  # loose adherence for natural variation
    ))

    # In production: stream to client's audio player, crossfade with previous soundscape
    return {"result": f"New soundscape playing: {params['energy_level']} energy"}
```

### Example conversation

> **Agent:** Hey! I'm your ambient workspace composer. Tell me what you're working on.
> **User:** I'm writing code and need deep focus. It's late at night.
> **Agent:** *(calls generate_soundscape: "Minimal ambient electronic, 60 BPM, soft synth pads with subtle granular textures, deeply meditative", "Late night rain on windows, very quiet distant thunder, soft keyboard typing sounds", "very_low")* Creating a late-night rain coding atmosphere for you.
> **User:** I'm fading. Give me some energy but don't break my flow.
> **Agent:** *(calls generate_soundscape: "Lo-fi hip hop beat, 85 BPM, warm bass, vinyl crackle, gentle Rhodes piano", "Coffee shop ambience, espresso machine, quiet chatter", "medium")* Shifting to a coffee shop vibe with a bit more rhythm.

### What makes this creative

The agent acts as a real-time sound designer that adapts to your mood through conversation. Each soundscape is unique -- never a pre-recorded loop. The voice interaction is deliberately minimal so it doesn't disrupt the focus it's trying to create.

---

## 7. Voice-Cloned Audiobook Narrator

**Skills:** `speech-to-text` + `text-to-speech` (with request stitching)

### Motivation

Audiobooks are expensive to produce because a narrator must record for hours. This workflow takes a short voice sample, creates a consistent narration across an entire book using request stitching to eliminate audio artifacts at chapter boundaries, and generates a timestamped chapter index.

### Implementation

```python
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import io, json, re

client = ElevenLabs()

def produce_audiobook(
    chapters: list[dict],  # [{"title": "Ch 1", "text": "..."}, ...]
    voice_id: str,
    output_dir: str = "audiobook",
) -> dict:
    """
    Produce a complete audiobook with chapter markers.

    Uses request stitching: each request knows what came before and after,
    so the voice flows naturally across chunk boundaries.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    CHUNK_SIZE = 2000  # characters per TTS request (stay under API limits)
    book_audio = AudioSegment.silent(0)
    chapter_markers = []

    voice_settings = VoiceSettings(
        stability=0.7,        # consistent for long-form
        similarity_boost=0.5, # natural variation
        style=0.0,            # neutral narration style
    )

    for ch_idx, chapter in enumerate(chapters):
        print(f"Generating chapter {ch_idx + 1}: {chapter['title']}")
        chapter_start_ms = len(book_audio)

        # Split chapter into chunks at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', chapter["text"])
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Generate audio for each chunk with stitching context
        for i, chunk_text in enumerate(chunks):
            previous = chunks[i - 1][-200:] if i > 0 else None  # last 200 chars of previous
            next_text = chunks[i + 1][:200] if i < len(chunks) - 1 else None  # first 200 chars of next

            kwargs = {
                "text": chunk_text,
                "voice_id": voice_id,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": voice_settings,
                "output_format": "mp3_44100_192",  # high quality for audiobook
            }
            if previous:
                kwargs["previous_text"] = previous
            if next_text:
                kwargs["next_text"] = next_text

            audio_iter = client.text_to_speech.convert(**kwargs)
            audio_bytes = b"".join(audio_iter)
            segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            book_audio += segment

        # Add a 2-second pause between chapters
        book_audio += AudioSegment.silent(2000)
        chapter_markers.append({
            "title": chapter["title"],
            "start_ms": chapter_start_ms,
            "start_formatted": f"{chapter_start_ms // 60000}:{(chapter_start_ms // 1000) % 60:02d}",
        })

    # Export final audiobook
    book_path = f"{output_dir}/audiobook.mp3"
    book_audio.export(book_path, format="mp3", bitrate="192k")

    # Generate chapter index
    index_path = f"{output_dir}/chapters.json"
    with open(index_path, "w") as f:
        json.dump(chapter_markers, f, indent=2)

    # Generate transcript with timestamps for accessibility
    with open(book_path, "rb") as f:
        transcript = client.speech_to_text.convert(
            file=f, model_id="scribe_v2", timestamps_granularity="word"
        )
    transcript_path = f"{output_dir}/transcript.json"
    with open(transcript_path, "w") as f:
        json.dump({"text": transcript.text, "words": [
            {"text": w.text, "start": w.start, "end": w.end}
            for w in transcript.words if w.type == "word"
        ]}, f, indent=2)

    return {"audio": book_path, "chapters": index_path, "transcript": transcript_path}
```

### What makes this creative

Request stitching is the key technique -- it eliminates the pops, pauses, and tonal shifts that plague naive chunk-by-chunk TTS. Combined with STT for a searchable transcript, the output is a professional audiobook with chapter markers and full-text search.

---

## 8. Customer Call Analytics Dashboard

**Skills:** `agents` (for live calls) + `speech-to-text` (for post-call analysis) + `text-to-speech` (for daily digest)

### Motivation

Customer support teams need both real-time voice agents and post-call analytics. This workflow deploys an AI agent to handle calls, then analyzes every completed conversation to extract sentiment, topics, resolution status, and compliance issues -- surfacing the results as a daily audio digest that a manager can listen to.

### Architecture

```
Inbound customer calls
      |
      v
  Agent (handles conversation)
      |  records conversation
      v
  Post-call webhook triggers analysis:
      |
      +---> STT: full transcript with timestamps + diarization
      +---> LLM: sentiment, topics, resolution, compliance
      +---> Database: store structured analytics
      |
      v
  Daily cron job:
      |
      +---> Aggregate day's analytics
      +---> LLM: generate narrative summary
      +---> TTS: produce audio digest
      |
      v
  "Daily Support Digest - Feb 16.mp3"
```

### Implementation (post-call analysis + daily digest)

```python
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import io, json
from datetime import date
from openai import OpenAI

el = ElevenLabs()
llm = OpenAI()

# --- Post-call analysis (runs after each call) ---
def analyze_call(recording_path: str, conversation_id: str) -> dict:
    with open(recording_path, "rb") as f:
        transcript = el.speech_to_text.convert(
            file=f, model_id="scribe_v2", diarize=True,
            timestamps_granularity="word",
        )

    # Build speaker-labeled text
    lines = []
    current_speaker, current_words = None, []
    for w in transcript.words:
        if w.type != "word":
            continue
        if w.speaker_id != current_speaker:
            if current_words:
                role = "Agent" if current_speaker == "speaker_0" else "Customer"
                lines.append(f"[{role}] {' '.join(current_words)}")
            current_speaker = w.speaker_id
            current_words = [w.text]
        else:
            current_words.append(w.text)
    if current_words:
        role = "Agent" if current_speaker == "speaker_0" else "Customer"
        lines.append(f"[{role}] {' '.join(current_words)}")

    labeled = "\n".join(lines)

    analysis = llm.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """Analyze this support call and return JSON:
{
  "sentiment": "positive|neutral|negative",
  "topics": ["topic1", "topic2"],
  "resolved": true/false,
  "resolution_summary": "one sentence",
  "customer_effort_score": 1-5,
  "compliance_flags": ["any policy violations"],
  "duration_seconds": estimated
}"""},
            {"role": "user", "content": labeled},
        ],
    )
    result = json.loads(analysis.choices[0].message.content)
    result["conversation_id"] = conversation_id
    result["transcript"] = labeled
    return result

# --- Daily digest (runs via cron) ---
def generate_daily_digest(call_analyses: list[dict], today: str) -> str:
    # Aggregate stats
    total = len(call_analyses)
    resolved = sum(1 for c in call_analyses if c.get("resolved"))
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for c in call_analyses:
        sentiments[c.get("sentiment", "neutral")] += 1

    all_topics = {}
    for c in call_analyses:
        for t in c.get("topics", []):
            all_topics[t] = all_topics.get(t, 0) + 1
    top_topics = sorted(all_topics.items(), key=lambda x: -x[1])[:5]

    flags = [f for c in call_analyses for f in c.get("compliance_flags", []) if f]

    # Generate narrative with LLM
    stats_text = f"""
Total calls: {total}
Resolved: {resolved}/{total} ({resolved/total*100:.0f}%)
Sentiment: {sentiments['positive']} positive, {sentiments['neutral']} neutral, {sentiments['negative']} negative
Top topics: {', '.join(f'{t[0]} ({t[1]})' for t in top_topics)}
Compliance flags: {len(flags)} total
"""

    narrative = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Write a 60-second audio briefing for a support team manager. "
                "Be concise, highlight what needs attention, and end with one actionable recommendation."
            )},
            {"role": "user", "content": stats_text},
        ],
    )
    script = narrative.choices[0].message.content

    # Generate audio digest
    audio = b"".join(el.text_to_speech.convert(
        text=f"Support digest for {today}. {script}",
        voice_id="onwK4e9ZLuTAKqWW03F9",  # Daniel - authoritative
        model_id="eleven_multilingual_v2",
    ))

    output_path = f"digest_{today}.mp3"
    with open(output_path, "wb") as f:
        f.write(audio)
    return output_path
```

### What makes this creative

Closes the loop from live AI calls to structured analytics to a consumable audio summary. The manager never has to read a dashboard -- they listen to a personalized briefing that highlights what actually needs their attention.

---

---

# Part 2: Workflows Using Extended ElevenLabs APIs

These workflows go beyond the repo's skills to use additional ElevenLabs API endpoints -- Text-to-Dialogue, Voice Design, Dubbing, Audio Isolation, and Agent WebSocket monitoring.

---

## 9. AI Radio Drama Producer (Text-to-Dialogue)

**APIs:** `Text-to-Dialogue` + `sound-effects` + `music`

### Motivation

The Text-to-Dialogue API (`/v1/text-to-dialogue`) with the Eleven v3 model generates multi-speaker conversations in a single API call -- complete with natural interruptions, overlapping cadence, and emotional cues. Combined with sound effects and music, this produces broadcast-quality radio dramas from a screenplay, with no manual audio editing.

The key advantage over Workflow #1 (AI Podcast Factory) is that Text-to-Dialogue generates **a single, naturally-paced audio file** with multiple speakers, rather than stitching separate TTS calls together. Speakers interrupt, pause, and react to each other.

### Architecture

```
Screenplay (tagged text)
      |
      v
  Script Parser --> dialogue segments + SFX/music cues
      |
      +---> Text-to-Dialogue API (multiple speakers, one call)
      +---> Sound Effects API (ambient + one-shots)
      +---> Music API (score for each scene)
      |
      v
  Audio Mixer (pydub) --> layers dialogue + SFX + music
      |
      v
  Final radio drama MP3
```

### Implementation

```python
import requests
import json
import re
import io
import os
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

client = ElevenLabs()
API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Voice cast ---
CAST = {
    "DETECTIVE": "onwK4e9ZLuTAKqWW03F9",  # Daniel - authoritative
    "SUSPECT":   "JBFqnCBsd6RMkjVDRZzb",  # George - nervous energy
    "NARRATOR":  "EXAVITQu4vr4xnSDxMaL",  # Sarah - calm
    "OFFICER":   "XB0fDUnXU5powFXDhCwa",  # Charlotte - professional
}

# --- Parse a screenplay into scenes ---
def parse_screenplay(text: str) -> list[dict]:
    """
    Parse screenplay format:
      [SCENE: Dark interrogation room]
      [MUSIC: Tense noir jazz, slow brushed drums]
      [SFX: Fluorescent light buzzing, chair scraping]
      DETECTIVE: Where were you last night?
      SUSPECT: [nervous laugh] I was... I was at home.
      DETECTIVE: [slams table] Don't lie to me!
      [SFX: Heavy door closing]
    """
    scenes = []
    current_scene = {"name": "", "dialogue": [], "music": None, "sfx": []}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if m := re.match(r"\[SCENE:\s*(.*?)\]", line):
            if current_scene["dialogue"]:
                scenes.append(current_scene)
            current_scene = {"name": m.group(1), "dialogue": [], "music": None, "sfx": []}
        elif m := re.match(r"\[MUSIC:\s*(.*?)\]", line):
            current_scene["music"] = m.group(1)
        elif m := re.match(r"\[SFX:\s*(.*?)\]", line):
            current_scene["sfx"].append(m.group(1))
        elif m := re.match(r"(\w+):\s*(.*)", line):
            character = m.group(1)
            dialogue_text = m.group(2)
            if character in CAST:
                current_scene["dialogue"].append({
                    "text": dialogue_text,
                    "voice_id": CAST[character],
                })

    if current_scene["dialogue"]:
        scenes.append(current_scene)
    return scenes

# --- Generate dialogue audio via Text-to-Dialogue API ---
def generate_dialogue(inputs: list[dict], seed: int = 42) -> AudioSegment:
    """
    Call POST /v1/text-to-dialogue with Eleven v3.
    Audio events in square brackets (e.g., [slams table]) are rendered
    naturally by the model -- no separate SFX call needed for inline cues.
    """
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-dialogue",
        headers={
            "xi-api-key": API_KEY,
            "Content-Type": "application/json",
        },
        params={"output_format": "mp3_44100_128"},
        json={
            "model_id": "eleven_v3",
            "inputs": inputs,
            "seed": seed,
            "settings": {"stability": 0.4},  # expressive for drama
        },
    )
    response.raise_for_status()
    return AudioSegment.from_mp3(io.BytesIO(response.content))

# --- Produce a full radio drama ---
def produce_radio_drama(screenplay: str, output_path: str = "drama.mp3"):
    scenes = parse_screenplay(screenplay)
    drama = AudioSegment.silent(0)

    for i, scene in enumerate(scenes):
        print(f"Producing scene {i+1}: {scene['name']}")

        # 1. Generate scene music (if any)
        music_seg = AudioSegment.silent(0)
        if scene["music"]:
            music_audio = b"".join(client.music.compose(
                prompt=scene["music"],
                music_length_ms=30000,
                force_instrumental=True,
            ))
            music_seg = AudioSegment.from_mp3(io.BytesIO(music_audio)) - 16  # quiet bed

        # 2. Generate ambient SFX
        sfx_seg = AudioSegment.silent(0)
        for sfx_prompt in scene["sfx"]:
            sfx_audio = b"".join(client.text_to_sound_effects.convert(
                text=sfx_prompt, duration_seconds=5.0, loop=True, prompt_influence=0.5,
            ))
            layer = AudioSegment.from_mp3(io.BytesIO(sfx_audio)) - 18
            sfx_seg = sfx_seg.overlay(layer) if len(sfx_seg) > 0 else layer

        # 3. Generate dialogue (single API call, natural multi-speaker pacing)
        dialogue_seg = generate_dialogue(scene["dialogue"])

        # 4. Mix: extend bed tracks to dialogue length, overlay
        target_len = len(dialogue_seg) + 2000
        if len(music_seg) > 0:
            while len(music_seg) < target_len:
                music_seg += music_seg
            music_seg = music_seg[:target_len].fade_out(2000)
        if len(sfx_seg) > 0:
            while len(sfx_seg) < target_len:
                sfx_seg += sfx_seg
            sfx_seg = sfx_seg[:target_len]

        scene_audio = AudioSegment.silent(target_len)
        if len(music_seg) > 0:
            scene_audio = scene_audio.overlay(music_seg)
        if len(sfx_seg) > 0:
            scene_audio = scene_audio.overlay(sfx_seg)
        scene_audio = scene_audio.overlay(dialogue_seg, position=500)

        drama += scene_audio + AudioSegment.silent(1500)  # gap between scenes

    drama = drama.fade_in(1000).fade_out(2000)
    drama.export(output_path, format="mp3")
    return output_path

# --- Example screenplay ---
SCREENPLAY = """
[SCENE: Dark interrogation room]
[MUSIC: Tense noir jazz, slow brushed drums, muted trumpet]
[SFX: Fluorescent light buzzing quietly]
NARRATOR: The interrogation room smelled of stale coffee and bad decisions.
DETECTIVE: Where were you last night between ten and midnight?
SUSPECT: [nervous laugh] I was at home. Watching TV. You can check.
DETECTIVE: [slams table] We already checked. Your neighbor says otherwise.
SUSPECT: [long pause] Okay... okay. I went out for a walk. That's not a crime.
[SFX: Heavy metal door slamming shut]
NARRATOR: The detective leaned back. He'd heard this story before.
"""

produce_radio_drama(SCREENPLAY)
```

### What makes this creative

Text-to-Dialogue renders inline audio events like `[slams table]` and `[nervous laugh]` as part of the speech generation -- no separate SFX pipeline needed for character actions. The Eleven v3 model handles natural turn-taking, pauses, and emotional shifts within a single API call.

---

## 10. Voice Casting Director (Voice Design API)

**APIs:** `Voice Design` (`/v1/text-to-voice/design` + `/v1/text-to-voice`) + `text-to-dialogue` + `agents`

### Motivation

When building voice applications (games, agents, audiobooks), you need voices that match specific characters -- but the voice library might not have what you need. This workflow uses the Voice Design API to generate custom voices from text descriptions, auditions them in a dialogue sample, and deploys the winner to a conversational agent. The entire casting process is automated.

### Architecture

```
Character brief (text description)
      |
      v
  Voice Design API (/v1/text-to-voice/design)
      |  returns 3 voice previews
      v
  Audition: Text-to-Dialogue with each candidate
      |  generates sample dialogue for comparison
      v
  Selection (LLM scores or user picks)
      |
      v
  Save Voice (/v1/text-to-voice)
      |
      v
  Deploy to Agent or TTS pipeline
```

### Implementation

```python
import requests
import json
import base64
import os
from elevenlabs.client import ElevenLabs

API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs()

# --- Step 1: Design voices from character descriptions ---
def design_voice(description: str, sample_text: str = None) -> list[dict]:
    """
    Generate 3 voice previews from a text description.
    Returns list of {generated_voice_id, audio_base64, duration_secs}.
    """
    body = {
        "voice_description": description,
        "model_id": "eleven_ttv_v3",
    }
    if sample_text:
        body["text"] = sample_text
    else:
        body["auto_generate_text"] = True

    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-voice/design",
        headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
        json=body,
    )
    response.raise_for_status()
    data = response.json()
    return data["previews"]

# --- Step 2: Audition voices in a dialogue ---
def audition_in_dialogue(
    candidate_voice_id: str,
    partner_voice_id: str,
    dialogue_lines: list[dict],
) -> bytes:
    """Generate a dialogue sample with the candidate voice."""
    # Replace placeholder with candidate voice
    inputs = []
    for line in dialogue_lines:
        voice = candidate_voice_id if line["role"] == "candidate" else partner_voice_id
        inputs.append({"text": line["text"], "voice_id": voice})

    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-dialogue",
        headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
        json={"model_id": "eleven_v3", "inputs": inputs},
    )
    response.raise_for_status()
    return response.content

# --- Step 3: Save the selected voice ---
def save_voice(generated_voice_id: str, name: str, description: str) -> str:
    """Save a designed voice to the library. Returns the permanent voice_id."""
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-voice",
        headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
        json={
            "voice_name": name,
            "voice_description": description,
            "generated_voice_id": generated_voice_id,
        },
    )
    response.raise_for_status()
    return response.json()["voice_id"]

# --- Full casting pipeline ---
def cast_character(
    character_name: str,
    character_description: str,
    audition_lines: list[dict],
    partner_voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # George as scene partner
) -> str:
    print(f"Casting: {character_name}")
    print(f"Description: {character_description}")

    # Generate 3 voice candidates
    previews = design_voice(
        description=character_description,
        sample_text=f"Hello, my name is {character_name}. I'm ready for my audition.",
    )

    # Audition each candidate
    audition_files = []
    for i, preview in enumerate(previews):
        print(f"  Auditioning candidate {i+1} ({preview['generated_voice_id'][:12]}...)")

        # Save preview audio for review
        audio_bytes = base64.b64decode(preview["audio_base_64"])
        preview_path = f"audition_{character_name}_{i+1}_preview.mp3"
        with open(preview_path, "wb") as f:
            f.write(audio_bytes)

        # Generate dialogue audition
        dialogue_audio = audition_in_dialogue(
            candidate_voice_id=preview["generated_voice_id"],
            partner_voice_id=partner_voice_id,
            dialogue_lines=audition_lines,
        )
        dialogue_path = f"audition_{character_name}_{i+1}_dialogue.mp3"
        with open(dialogue_path, "wb") as f:
            f.write(dialogue_audio)

        audition_files.append({
            "candidate": i + 1,
            "generated_voice_id": preview["generated_voice_id"],
            "preview": preview_path,
            "dialogue": dialogue_path,
            "duration": preview["duration_secs"],
        })

    # In production: use LLM to score or present to user for selection
    # For now, select candidate 1
    selected = audition_files[0]
    print(f"  Selected candidate {selected['candidate']}")

    # Save to library
    voice_id = save_voice(
        generated_voice_id=selected["generated_voice_id"],
        name=character_name,
        description=character_description,
    )
    print(f"  Saved as voice_id: {voice_id}")
    return voice_id

# --- Example: Cast a villain for a game ---
villain_voice_id = cast_character(
    character_name="Lord Vexar",
    character_description=(
        "A deep, gravelly male voice with a menacing British accent. "
        "Speaks slowly and deliberately, as if savoring every word. "
        "Age 50-60. Think theatrical villain with restrained intensity."
    ),
    audition_lines=[
        {"role": "candidate", "text": "You think you can stop me? How delightfully naive."},
        {"role": "partner", "text": "This ends now, Vexar."},
        {"role": "candidate", "text": "[dark chuckle] Oh, my dear hero. This is only the beginning."},
    ],
)

# Deploy to an agent
agent = client.conversational_ai.agents.create(
    name="Lord Vexar",
    conversation_config={
        "agent": {"first_message": "You dare approach my throne?", "language": "en"},
        "tts": {"voice_id": villain_voice_id, "model_id": "eleven_flash_v2_5"},
    },
    prompt={
        "prompt": "You are Lord Vexar, a theatrical villain. Stay in character.",
        "llm": "gpt-4o-mini", "temperature": 0.9,
    },
)
```

### What makes this creative

Automates the entire voice casting process -- from description to audition to deployment. The Voice Design API generates candidates from a character brief, Text-to-Dialogue auditions them in context (not isolation), and the winner gets saved to the library and deployed to an agent in one pipeline.

---

## 11. Video Localization Pipeline (Dubbing + STT + TTS)

**APIs:** `Dubbing` (`/v1/dubbing`) + `speech-to-text` + `text-to-speech` + `Audio Isolation`

### Motivation

Content creators need to localize videos for global audiences. The ElevenLabs Dubbing API handles the heavy lifting (translation + voice-cloned dubbing), but a complete pipeline also needs: quality assurance transcription of the dubbed output, isolated clean audio for subtitle generation, and promotional clips in each target language.

### Architecture

```
Source video (English)
      |
      v
  Dubbing API (/v1/dubbing)
      |  target_lang for each market
      +---> Spanish dub
      +---> French dub
      +---> Japanese dub
      |
      v
  For each dubbed output:
      |
      +---> Audio Isolation (clean up any artifacts)
      +---> STT (generate subtitles with timestamps)
      +---> TTS (generate promotional trailer narration)
      |
      v
  Localized package per language:
    - Dubbed video
    - Clean audio track
    - SRT subtitle file
    - Promotional audio clip
```

### Implementation

```python
import requests
import time
import json
import os
from elevenlabs.client import ElevenLabs

API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs()

# --- Step 1: Submit dubbing jobs ---
def submit_dub(
    source_url: str,
    target_lang: str,
    source_lang: str = "en",
    name: str = None,
) -> dict:
    """Submit a video for dubbing. Returns dubbing_id and expected duration."""
    response = requests.post(
        "https://api.elevenlabs.io/v1/dubbing",
        headers={"xi-api-key": API_KEY},
        data={
            "source_url": source_url,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "name": name or f"dub_{target_lang}",
            "num_speakers": 0,  # auto-detect
            "highest_resolution": True,
        },
    )
    response.raise_for_status()
    return response.json()  # {"dubbing_id": "...", "expected_duration_sec": ...}

def wait_for_dub(dubbing_id: str, poll_interval: int = 15) -> str:
    """Poll until dubbing is complete. Returns status."""
    while True:
        response = requests.get(
            f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}",
            headers={"xi-api-key": API_KEY},
        )
        response.raise_for_status()
        status = response.json().get("status", "unknown")
        if status == "dubbed":
            return status
        elif status in ("failed", "error"):
            raise RuntimeError(f"Dubbing failed: {response.json()}")
        print(f"  Status: {status}, waiting {poll_interval}s...")
        time.sleep(poll_interval)

def download_dub(dubbing_id: str, language: str, output_path: str) -> str:
    """Download the dubbed video/audio."""
    response = requests.get(
        f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{language}",
        headers={"xi-api-key": API_KEY},
    )
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

# --- Step 2: Clean audio with Audio Isolation ---
def isolate_audio(audio_path: str, output_path: str) -> str:
    """Remove background noise/artifacts from dubbed audio."""
    with open(audio_path, "rb") as f:
        response = requests.post(
            "https://api.elevenlabs.io/v1/audio-isolation",
            headers={"xi-api-key": API_KEY},
            files={"audio": f},
        )
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

# --- Step 3: Generate subtitles from dubbed audio ---
def generate_subtitles(audio_path: str, language: str) -> list[dict]:
    """Transcribe dubbed audio and generate SRT-compatible timestamps."""
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f, model_id="scribe_v2",
            timestamps_granularity="word",
            language_code=language,
        )

    # Group words into subtitle segments (~8 words each)
    subtitles = []
    words = [w for w in result.words if w.type == "word"]
    for i in range(0, len(words), 8):
        segment = words[i:i+8]
        subtitles.append({
            "index": len(subtitles) + 1,
            "start": segment[0].start,
            "end": segment[-1].end,
            "text": " ".join(w.text for w in segment),
        })
    return subtitles

def write_srt(subtitles: list[dict], output_path: str):
    """Write subtitles in SRT format."""
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{format_time(sub['start'])} --> {format_time(sub['end'])}\n")
            f.write(f"{sub['text']}\n\n")

# --- Step 4: Generate promotional narration ---
def generate_promo(text: str, language: str, output_path: str):
    """Generate a promotional clip narration in the target language."""
    audio = b"".join(client.text_to_speech.convert(
        text=text,
        voice_id="EXAVITQu4vr4xnSDxMaL",  # Sarah
        model_id="eleven_multilingual_v2",
        language_code=language,
    ))
    with open(output_path, "wb") as f:
        f.write(audio)

# --- Full pipeline ---
def localize_video(
    source_url: str,
    target_languages: dict[str, str],  # {"es": "Spanish", "fr": "French", ...}
    promo_texts: dict[str, str],       # {"es": "Disponible ahora...", ...}
    output_dir: str = "localized",
):
    os.makedirs(output_dir, exist_ok=True)

    for lang_code, lang_name in target_languages.items():
        print(f"\n=== Localizing to {lang_name} ({lang_code}) ===")
        lang_dir = f"{output_dir}/{lang_code}"
        os.makedirs(lang_dir, exist_ok=True)

        # Submit dub
        result = submit_dub(source_url, target_lang=lang_code, name=f"Video - {lang_name}")
        dubbing_id = result["dubbing_id"]
        print(f"  Dubbing submitted: {dubbing_id} (est. {result['expected_duration_sec']}s)")

        # Wait for completion
        wait_for_dub(dubbing_id)
        print(f"  Dubbing complete!")

        # Download
        dub_path = download_dub(dubbing_id, lang_code, f"{lang_dir}/dubbed.mp3")

        # Clean audio
        clean_path = isolate_audio(dub_path, f"{lang_dir}/clean_audio.mp3")
        print(f"  Audio isolated")

        # Generate subtitles
        subtitles = generate_subtitles(clean_path, lang_code)
        write_srt(subtitles, f"{lang_dir}/subtitles.srt")
        print(f"  Subtitles generated ({len(subtitles)} segments)")

        # Promo narration
        if lang_code in promo_texts:
            generate_promo(promo_texts[lang_code], lang_code, f"{lang_dir}/promo.mp3")
            print(f"  Promo narration generated")

# --- Example ---
localize_video(
    source_url="https://example.com/product-demo.mp4",
    target_languages={"es": "Spanish", "fr": "French", "ja": "Japanese"},
    promo_texts={
        "es": "Descubre nuestro nuevo producto. Disponible ahora en tu idioma.",
        "fr": "Decouvrez notre nouveau produit. Disponible maintenant dans votre langue.",
        "ja": "新製品をご覧ください。あなたの言語でご利用いただけます。",
    },
)
```

### What makes this creative

Chains four APIs into a complete localization factory: Dubbing for translation + voice cloning, Audio Isolation to clean artifacts, STT for auto-generated subtitles, and TTS for promotional audio -- producing a complete localized package per language from a single source video.

---

## 12. Podcast Cleanup & Remaster (Audio Isolation + STT + Text-to-Dialogue)

**APIs:** `Audio Isolation` + `speech-to-text` + `Text-to-Dialogue`

### Motivation

Many podcasts and interviews are recorded in noisy environments -- cafes, conference floors, home offices with barking dogs. This workflow takes a noisy recording, isolates the speech, transcribes it with speaker labels, then re-records the entire conversation using Text-to-Dialogue to produce a studio-quality version with the original words but crystal-clear audio. Optionally preserves the original voices via voice cloning.

### Architecture

```
Noisy podcast recording
      |
      v
  Audio Isolation (/v1/audio-isolation)
      |  cleaned audio
      v
  STT with diarization (scribe_v2)
      |  speaker-labeled transcript
      v
  Voice matching (assign voice IDs to speakers)
      |
      v
  Text-to-Dialogue (/v1/text-to-dialogue)
      |  re-records the conversation with natural pacing
      v
  Studio-quality podcast + original transcript
```

### Implementation

```python
import requests
import json
import io
import os
from elevenlabs.client import ElevenLabs

API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs()

def remaster_podcast(
    noisy_audio_path: str,
    speaker_voices: dict[str, str],  # {"speaker_0": "voice_id_1", "speaker_1": "voice_id_2"}
    output_path: str = "remastered.mp3",
) -> dict:
    """
    Take a noisy podcast, clean it, transcribe it, and re-record it
    with Text-to-Dialogue for studio quality.
    """

    # 1. Isolate speech from background noise
    print("Step 1: Isolating speech...")
    with open(noisy_audio_path, "rb") as f:
        isolation_response = requests.post(
            "https://api.elevenlabs.io/v1/audio-isolation",
            headers={"xi-api-key": API_KEY},
            files={"audio": f},
        )
    isolation_response.raise_for_status()
    clean_audio_path = noisy_audio_path.replace(".mp3", "_clean.mp3")
    with open(clean_audio_path, "wb") as f:
        f.write(isolation_response.content)

    # 2. Transcribe with speaker diarization
    print("Step 2: Transcribing with speaker labels...")
    with open(clean_audio_path, "rb") as f:
        transcript = client.speech_to_text.convert(
            file=f, model_id="scribe_v2",
            diarize=True, timestamps_granularity="word",
        )

    # 3. Build speaker-segmented dialogue
    print("Step 3: Building dialogue segments...")
    segments = []
    current_speaker = None
    current_text = []

    for word in transcript.words:
        if word.type != "word":
            continue
        if word.speaker_id != current_speaker:
            if current_text and current_speaker:
                segments.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                })
            current_speaker = word.speaker_id
            current_text = [word.text]
        else:
            current_text.append(word.text)

    if current_text and current_speaker:
        segments.append({"speaker": current_speaker, "text": " ".join(current_text)})

    # 4. Re-record with Text-to-Dialogue
    print(f"Step 4: Re-recording {len(segments)} segments with Text-to-Dialogue...")

    # Text-to-Dialogue has a max of 10 unique voices and reasonable text limits
    # Process in chunks if needed
    CHUNK_SIZE = 20  # segments per API call
    all_audio = []

    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i:i + CHUNK_SIZE]
        inputs = []
        for seg in chunk:
            voice_id = speaker_voices.get(seg["speaker"], list(speaker_voices.values())[0])
            inputs.append({"text": seg["text"], "voice_id": voice_id})

        response = requests.post(
            "https://api.elevenlabs.io/v1/text-to-dialogue",
            headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
            json={
                "model_id": "eleven_v3",
                "inputs": inputs,
                "settings": {"stability": 0.5},
            },
        )
        response.raise_for_status()
        all_audio.append(response.content)

    # 5. Concatenate and export
    from pydub import AudioSegment as AS
    final = AS.silent(0)
    for audio_bytes in all_audio:
        segment = AS.from_mp3(io.BytesIO(audio_bytes))
        final += segment + AS.silent(200)

    final.export(output_path, format="mp3")
    print(f"Remastered podcast saved to {output_path}")

    return {
        "output": output_path,
        "original_transcript": transcript.text,
        "segments": len(segments),
        "speakers": list(set(s["speaker"] for s in segments)),
    }

# --- Example ---
result = remaster_podcast(
    noisy_audio_path="noisy_interview.mp3",
    speaker_voices={
        "speaker_0": "onwK4e9ZLuTAKqWW03F9",  # Daniel (interviewer)
        "speaker_1": "JBFqnCBsd6RMkjVDRZzb",  # George (guest)
    },
)
```

### What makes this creative

Three APIs in sequence transform unusable audio into broadcast quality: isolation removes noise, STT recovers the words with speaker labels, and Text-to-Dialogue re-records the conversation with natural multi-speaker pacing. The original content is preserved but the audio quality is transformed.

---

## 13. Live Agent QA Monitor (Agent WebSocket + STT + TTS)

**APIs:** `Agent WebSocket monitoring` (`/v1/convai/conversations/{id}/monitor`) + `agents` + `speech-to-text` + `text-to-speech`

### Motivation

When deploying AI voice agents at scale (customer service, sales), supervisors need real-time visibility into live calls with the ability to intervene -- whisper coaching to the agent, barging into the call, or triggering a human takeover. The Agent WebSocket monitoring API provides exactly this, and combined with STT and TTS, enables a complete supervisor dashboard.

### Architecture

```
Live agent call in progress
      |
      v
  WebSocket monitor (/v1/convai/conversations/{id}/monitor)
      |  real-time events: user_transcript, agent_response, vad_score
      |
      +---> Live transcript display
      +---> Sentiment analysis (LLM on each turn)
      +---> Alert triggers (compliance, escalation keywords)
      |
      v
  Supervisor actions:
      +---> Barge-in (inject message into conversation)
      +---> End call
      +---> Human takeover
      +---> Post-call: TTS summary for supervisor review
```

### Implementation

```python
import asyncio
import json
import os
import websockets
from datetime import datetime
from elevenlabs.client import ElevenLabs
from openai import OpenAI

API_KEY = os.getenv("ELEVENLABS_API_KEY")
el_client = ElevenLabs()
llm = OpenAI()

# --- Real-time call monitor ---
class CallMonitor:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.transcript = []
        self.alerts = []
        self.sentiment_history = []

    async def connect(self):
        """Connect to the agent monitoring WebSocket."""
        uri = (
            f"wss://api.elevenlabs.io/v1/convai/conversations"
            f"/{self.conversation_id}/monitor"
        )
        headers = {"xi-api-key": API_KEY}

        async with websockets.connect(uri, extra_headers=headers) as ws:
            print(f"[MONITOR] Connected to conversation {self.conversation_id}")

            # Subscribe to events
            await ws.send(json.dumps({
                "type": "subscribe",
                "events": [
                    "user_transcript",
                    "agent_response",
                    "vad_score",
                    "client_tool_call",
                    "interruption",
                ],
            }))

            async for message in ws:
                event = json.loads(message)
                await self.handle_event(event, ws)

    async def handle_event(self, event: dict, ws):
        event_type = event.get("type", "")
        timestamp = datetime.now().strftime("%H:%M:%S")

        if event_type == "user_transcript":
            text = event.get("text", "")
            self.transcript.append({"role": "user", "text": text, "time": timestamp})
            print(f"[{timestamp}] CUSTOMER: {text}")
            await self.analyze_turn(text, "customer")

        elif event_type == "agent_response":
            text = event.get("text", "")
            self.transcript.append({"role": "agent", "text": text, "time": timestamp})
            print(f"[{timestamp}] AGENT: {text}")
            await self.analyze_turn(text, "agent")

        elif event_type == "interruption":
            print(f"[{timestamp}] ** Customer interrupted agent **")

        elif event_type == "client_tool_call":
            tool = event.get("tool_name", "unknown")
            print(f"[{timestamp}] TOOL CALL: {tool}")

        elif event_type == "vad_score":
            score = event.get("score", 0)
            if score > 0.8:
                print(f"[{timestamp}] [VAD: active speech detected]")

    async def analyze_turn(self, text: str, speaker: str):
        """Analyze each turn for sentiment and compliance flags."""
        # Check for escalation keywords
        ESCALATION_KEYWORDS = [
            "supervisor", "manager", "lawsuit", "cancel", "angry",
            "unacceptable", "report", "complaint", "lawyer",
        ]
        text_lower = text.lower()
        for keyword in ESCALATION_KEYWORDS:
            if keyword in text_lower:
                alert = f"ESCALATION KEYWORD detected: '{keyword}' from {speaker}"
                self.alerts.append(alert)
                print(f"  !! ALERT: {alert}")
                break

    # --- Supervisor control commands ---
    async def send_control(self, ws, command: str, **kwargs):
        """Send a control command to the live conversation."""
        msg = {"type": command, **kwargs}
        await ws.send(json.dumps(msg))
        print(f"[CONTROL] Sent: {command}")

    async def barge_in(self, ws, message: str):
        """Inject a message into the agent's context."""
        await self.send_control(ws, "contextual_update", text=message)

    async def end_call(self, ws):
        """Force-end the current call."""
        await self.send_control(ws, "end_conversation")

    # --- Post-call summary ---
    def generate_summary(self) -> str:
        """Generate a post-call audio summary for the supervisor."""
        transcript_text = "\n".join(
            f"[{t['role'].upper()}] {t['text']}" for t in self.transcript
        )

        analysis = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Summarize this customer call in 3 sentences for a supervisor. "
                    "Note: sentiment, whether the issue was resolved, any concerning moments."
                )},
                {"role": "user", "content": transcript_text},
            ],
        )
        summary_text = analysis.choices[0].message.content

        # Generate audio summary
        audio = b"".join(el_client.text_to_speech.convert(
            text=f"Call summary. {summary_text}",
            voice_id="onwK4e9ZLuTAKqWW03F9",
            model_id="eleven_flash_v2_5",
        ))

        output_path = f"call_summary_{self.conversation_id[:8]}.mp3"
        with open(output_path, "wb") as f:
            f.write(audio)
        return output_path

# --- Usage ---
async def monitor_call(conversation_id: str):
    monitor = CallMonitor(conversation_id)
    try:
        await monitor.connect()
    except websockets.exceptions.ConnectionClosed:
        print("[MONITOR] Call ended")
    finally:
        summary_path = monitor.generate_summary()
        print(f"\nPost-call summary: {summary_path}")
        if monitor.alerts:
            print(f"Alerts triggered: {len(monitor.alerts)}")
            for alert in monitor.alerts:
                print(f"  - {alert}")

# asyncio.run(monitor_call("conversation-id-here"))
```

### What makes this creative

The Agent WebSocket monitoring API enables a live supervision layer over AI phone agents -- real-time transcript streaming, keyword-based escalation alerts, and supervisor intervention (barge-in, takeover, end call). Post-call, the same pipeline generates an audio summary. This is the missing piece for deploying voice agents at enterprise scale.

---

## 14. Pronunciation-Aware Technical Narrator (Pronunciation Dictionaries + TTS + STT)

**APIs:** `Pronunciation Dictionaries` + `text-to-speech` + `speech-to-text` + `Text-to-Dialogue`

### Motivation

Technical content (medical, legal, scientific, brand-specific) is full of terms that TTS models mispronounce -- drug names, API endpoints, brand neologisms, acronyms. This workflow creates a pronunciation dictionary from a glossary, generates audio with correct pronunciation, then validates the output with STT to catch any remaining errors. For dialogue content (e.g., doctor-patient conversations), it feeds the dictionary into Text-to-Dialogue.

### Architecture

```
Technical glossary (term -> pronunciation)
      |
      v
  Create Pronunciation Dictionary (/v1/pronunciation-dictionaries)
      |  dictionary_id + version_id
      v
  TTS with dictionary attached
      |  generates audio with correct pronunciation
      v
  STT validation pass
      |  transcribes output, compares against expected terms
      v
  Report: which terms were pronounced correctly vs. need adjustment
```

### Implementation

```python
import requests
import json
import os
from elevenlabs.client import ElevenLabs

API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs()

# --- Step 1: Create a pronunciation dictionary ---
def create_pronunciation_dictionary(
    name: str,
    entries: list[dict],  # [{"term": "GLP-1", "pronunciation": "G L P one"}, ...]
) -> dict:
    """
    Create a pronunciation dictionary using phoneme or alias rules.

    Entries can use:
    - alias: {"term": "CRISPR", "alias": "krisper"}
    - IPA: {"term": "acetaminophen", "ipa": "əˌsiːtəˈmɪnəfɪn"}
    """
    # Build the XML rules file
    rules = ['<?xml version="1.0" encoding="UTF-8"?>']
    rules.append("<lexicon>")
    for entry in entries:
        if "ipa" in entry:
            rules.append(f'  <lexeme>')
            rules.append(f'    <grapheme>{entry["term"]}</grapheme>')
            rules.append(f'    <phoneme alphabet="ipa">{entry["ipa"]}</phoneme>')
            rules.append(f'  </lexeme>')
        elif "alias" in entry:
            rules.append(f'  <lexeme>')
            rules.append(f'    <grapheme>{entry["term"]}</grapheme>')
            rules.append(f'    <alias>{entry["alias"]}</alias>')
            rules.append(f'  </lexeme>')
    rules.append("</lexicon>")

    rules_content = "\n".join(rules)

    response = requests.post(
        "https://api.elevenlabs.io/v1/pronunciation-dictionaries/add-from-file",
        headers={"xi-api-key": API_KEY},
        files={"file": ("dictionary.pls", rules_content, "application/xml")},
        data={"name": name, "description": f"Technical terms for {name}"},
    )
    response.raise_for_status()
    data = response.json()
    return {
        "dictionary_id": data["id"],
        "version_id": data["version_id"],
        "name": name,
    }

# --- Step 2: Generate audio with dictionary ---
def narrate_with_dictionary(
    text: str,
    dictionary: dict,
    voice_id: str = "onwK4e9ZLuTAKqWW03F9",
    output_path: str = "narration.mp3",
) -> str:
    """Generate TTS audio with pronunciation dictionary applied."""
    audio = b"".join(client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        pronunciation_dictionary_locators=[{
            "pronunciation_dictionary_id": dictionary["dictionary_id"],
            "version_id": dictionary["version_id"],
        }],
    ))
    with open(output_path, "wb") as f:
        f.write(audio)
    return output_path

# --- Step 3: Validate pronunciation with STT ---
def validate_pronunciation(
    audio_path: str,
    expected_terms: list[str],
) -> dict:
    """Transcribe the audio and check if expected terms appear correctly."""
    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f, model_id="scribe_v2",
            keyterms=expected_terms,  # bias STT toward these terms
        )

    transcript = result.text.lower()
    report = {"transcript": result.text, "results": []}

    for term in expected_terms:
        found = term.lower() in transcript
        report["results"].append({
            "term": term,
            "found_in_transcript": found,
            "status": "PASS" if found else "CHECK",
        })

    passed = sum(1 for r in report["results"] if r["status"] == "PASS")
    report["summary"] = f"{passed}/{len(expected_terms)} terms validated"
    return report

# --- Step 4: Dialogue with pronunciation dictionary ---
def technical_dialogue_with_dictionary(
    dialogue_inputs: list[dict],
    dictionary: dict,
    output_path: str = "technical_dialogue.mp3",
) -> str:
    """Generate a multi-speaker technical dialogue with pronunciation rules."""
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-dialogue",
        headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
        json={
            "model_id": "eleven_v3",
            "inputs": dialogue_inputs,
            "pronunciation_dictionary_locators": [{
                "pronunciation_dictionary_id": dictionary["dictionary_id"],
                "version_id": dictionary["version_id"],
            }],
        },
    )
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

# --- Full pipeline example: Medical narration ---
# 1. Define glossary
medical_glossary = [
    {"term": "GLP-1", "alias": "G L P one"},
    {"term": "HbA1c", "alias": "hemoglobin A one C"},
    {"term": "semaglutide", "alias": "sem-ah-GLOO-tide"},
    {"term": "tirzepatide", "alias": "ter-ZEP-ah-tide"},
    {"term": "SGLT2", "alias": "S G L T two"},
    {"term": "metformin", "ipa": "mɛtˈfɔːrmɪn"},
]

# 2. Create dictionary
dictionary = create_pronunciation_dictionary("Endocrinology Terms", medical_glossary)
print(f"Dictionary created: {dictionary['dictionary_id']}")

# 3. Generate narration
narration_text = (
    "The patient was started on semaglutide, a GLP-1 receptor agonist, "
    "after their HbA1c remained above target on metformin alone. "
    "The care team also considered tirzepatide and an SGLT2 inhibitor "
    "as alternative options."
)
audio_path = narrate_with_dictionary(narration_text, dictionary)
print(f"Narration generated: {audio_path}")

# 4. Validate
report = validate_pronunciation(
    audio_path,
    expected_terms=["semaglutide", "GLP-1", "HbA1c", "metformin", "tirzepatide", "SGLT2"],
)
print(f"Validation: {report['summary']}")
for r in report["results"]:
    print(f"  {r['term']}: {r['status']}")

# 5. Generate a doctor-patient dialogue with correct pronunciation
dialogue_path = technical_dialogue_with_dictionary(
    dialogue_inputs=[
        {"text": "Your HbA1c is still at 8.2. I think we should add semaglutide.", "voice_id": "onwK4e9ZLuTAKqWW03F9"},
        {"text": "Is that the GLP-1 medication? I've heard about tirzepatide too.", "voice_id": "EXAVITQu4vr4xnSDxMaL"},
        {"text": "Both are good options. Semaglutide has more long-term data. We'll keep the metformin as well.", "voice_id": "onwK4e9ZLuTAKqWW03F9"},
    ],
    dictionary=dictionary,
)
print(f"Technical dialogue generated: {dialogue_path}")
```

### What makes this creative

Closes the pronunciation quality loop: dictionary ensures correct TTS pronunciation, STT validation catches failures, and keyterm prompting biases the transcription toward the expected terms for accurate comparison. Works with both single-speaker TTS and multi-speaker Text-to-Dialogue.

---

## Summary: All Workflow Combinations

| # | Workflow | TTS | STT | SFX | Music | Agents | CLI | Dialogue | Voice Design | Dubbing | Isolation | Monitor | Pron. Dict |
|---|----------|-----|-----|-----|-------|--------|-----|----------|-------------|---------|-----------|---------|------------|
| 1 | AI Podcast Factory | x | x | x | x | | | | | | | | |
| 2 | Live Translation Booth | x | x | | | | | | | | | | |
| 3 | Audio Adventure Engine | x | | x | x | x | | | | | | | |
| 4 | Mood Journal | x | x | x | x | | | | | | | | |
| 5 | Meeting Intelligence | x | | | | | x | | | | | | |
| 6 | Ambient Workspace | | | x | x | x | | | | | | | |
| 7 | Audiobook Narrator | x | x | | | | | | | | | | |
| 8 | Call Analytics Dashboard | x | x | | | x | | | | | | | |
| 9 | AI Radio Drama | | | x | x | | | x | | | | | |
| 10 | Voice Casting Director | | | | | x | | x | x | | | | |
| 11 | Video Localization | x | x | | | | | | | x | x | | |
| 12 | Podcast Remaster | | x | | | | | x | | | x | | |
| 13 | Live Agent QA Monitor | x | | | | x | | | | | | x | |
| 14 | Technical Narrator | x | x | | | | | x | | | | | x |

**API coverage:** Every ElevenLabs API product appears in at least one workflow. The new workflows (#9-14) specifically showcase the APIs that go beyond the repo's skills.
