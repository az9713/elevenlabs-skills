# Creative Multi-Skill Workflows

Eight production-ready workflows that combine ElevenLabs skills in unexpected ways. Each workflow lists the skills it chains together, explains the motivation, and provides enough implementation detail to build it.

**Skills referenced:** setup-api-key, text-to-speech, speech-to-text, sound-effects, music, agents, elevenlabs-transcribe

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

## Summary: Skill Combinations Used

| Workflow | TTS | STT | SFX | Music | Agents | Transcribe CLI |
|----------|-----|-----|-----|-------|--------|----------------|
| AI Podcast Factory | x | x | x | x | | |
| Live Translation Booth | x | x | | | | |
| Audio Adventure Engine | x | | x | x | x | |
| Mood Journal | x | x | x | x | | |
| Meeting Intelligence | x | | | | | x |
| Ambient Workspace | | | x | x | x | |
| Audiobook Narrator | x | x | | | | |
| Call Analytics Dashboard | x | x | | | x | |

Every skill appears in at least two workflows. The most versatile skill is `text-to-speech` (6/8 workflows), followed by `speech-to-text` (5/8).
