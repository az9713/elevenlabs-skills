"""
Immersive Audio Adventure Engine
================================
A voice-controlled text adventure where you speak into your mic, an AI Dungeon
Master narrates back, and background music + sound effects are generated
dynamically using the ElevenLabs Music and Sound Effects APIs.

Architecture:
  - ElevenLabs Conversation SDK handles mic capture, WebSocket, and agent voice
  - Our code layers music/SFX on top via pygame.mixer when the agent calls
    client tools (set_scene, play_effect)

Usage:
  python adventure.py
"""

import json
import os
import sys
import threading
from hashlib import md5
from pathlib import Path

import pygame
from dotenv import load_dotenv
from elevenlabs import (
    AgentConfig,
    ConversationalConfig,
    PromptAgentApiModelOutput,
    TtsConversationalConfigOutput,
)
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import ClientTools, Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    print("Error: ELEVENLABS_API_KEY not set. Copy .env from ../voice-migration/")
    sys.exit(1)

client = ElevenLabs(api_key=API_KEY)

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "agent_config.json"
CACHE_DIR = SCRIPT_DIR / "audio_cache"
CACHE_DIR.mkdir(exist_ok=True)

# George voice - authoritative British male, perfect for a Dungeon Master
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

SYSTEM_PROMPT = """\
You are a Dungeon Master for an immersive audio text adventure game.

Your role:
- Narrate a rich fantasy adventure with vivid descriptions
- Present the player with meaningful choices after each scene
- Track game state (inventory, health, location) and reference it naturally
- Keep responses concise (2-4 sentences of narration + choices)

IMPORTANT - You have two tools you MUST use to create atmosphere:

1. **set_scene** - Call this whenever the scene changes (new location, mood shift,
   time change). Provide a "mood" (e.g. "mysterious", "triumphant", "dark") and
   an "environment" description for ambient sound (e.g. "deep cave with dripping
   water", "bustling medieval tavern").

2. **play_effect** - Call this for dramatic moments: combat hits, door opening,
   treasure found, monster roaring, spell casting, etc. Provide a short
   "description" of the sound.

Call set_scene at the START of the adventure and whenever the location/mood
changes. Call play_effect during action moments. You can call both in the same
turn if appropriate.

Begin by welcoming the player and setting the opening scene of the adventure.
"""

# ---------------------------------------------------------------------------
# Pygame mixer for layered audio
# ---------------------------------------------------------------------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
pygame.mixer.set_num_channels(4)

CH_MUSIC = pygame.mixer.Channel(0)      # Background music (looped)
CH_AMBIENT = pygame.mixer.Channel(1)    # Ambient SFX (looped)
CH_EFFECT = pygame.mixer.Channel(2)     # One-shot sound effects
# Channel 3 reserved for future use

CH_MUSIC.set_volume(0.3)
CH_AMBIENT.set_volume(0.4)
CH_EFFECT.set_volume(0.7)

# Simple cache: hash of prompt -> file path
audio_cache: dict[str, Path] = {}


def _cache_key(prefix: str, text: str) -> str:
    return f"{prefix}_{md5(text.encode()).hexdigest()[:12]}"


def _get_cached(key: str) -> Path | None:
    if key in audio_cache and audio_cache[key].exists():
        return audio_cache[key]
    path = CACHE_DIR / f"{key}.mp3"
    if path.exists():
        audio_cache[key] = path
        return path
    return None


def _save_to_cache(key: str, audio_bytes: bytes) -> Path:
    path = CACHE_DIR / f"{key}.mp3"
    path.write_bytes(audio_bytes)
    audio_cache[key] = path
    return path


# ---------------------------------------------------------------------------
# Audio generation helpers (run in background threads)
# ---------------------------------------------------------------------------

def _generate_and_play_music(mood: str, environment: str):
    """Generate background music and ambient SFX for a scene."""
    prompt = f"Atmospheric {mood} fantasy adventure background music, instrumental, "
    prompt += f"setting: {environment}"

    cache_key = _cache_key("music", prompt)
    cached = _get_cached(cache_key)

    if cached:
        print(f"  [Music] Playing cached: {mood}")
    else:
        print(f"  [Music] Generating: {mood} / {environment}")
        try:
            audio_iter = client.music.compose(
                prompt=prompt,
                music_length_ms=30_000,
                force_instrumental=True,
            )
            audio_bytes = b"".join(audio_iter)
            cached = _save_to_cache(cache_key, audio_bytes)
            print(f"  [Music] Generated ({len(audio_bytes) // 1024}KB)")
        except Exception as e:
            print(f"  [Music] Error: {e}")
            return

    try:
        sound = pygame.mixer.Sound(str(cached))
        CH_MUSIC.play(sound, loops=-1, fade_ms=2000)
    except Exception as e:
        print(f"  [Music] Playback error: {e}")

    # Ambient SFX
    ambient_prompt = f"{environment} ambient sounds, subtle background atmosphere"
    amb_key = _cache_key("ambient", ambient_prompt)
    amb_cached = _get_cached(amb_key)

    if amb_cached:
        print(f"  [Ambient] Playing cached")
    else:
        print(f"  [Ambient] Generating: {environment}")
        try:
            sfx_iter = client.text_to_sound_effects.convert(
                text=ambient_prompt,
                duration_seconds=10.0,
                prompt_influence=0.4,
            )
            sfx_bytes = b"".join(sfx_iter)
            amb_cached = _save_to_cache(amb_key, sfx_bytes)
            print(f"  [Ambient] Generated ({len(sfx_bytes) // 1024}KB)")
        except Exception as e:
            print(f"  [Ambient] Error: {e}")
            return

    try:
        sound = pygame.mixer.Sound(str(amb_cached))
        CH_AMBIENT.play(sound, loops=-1, fade_ms=1000)
    except Exception as e:
        print(f"  [Ambient] Playback error: {e}")


def _generate_and_play_effect(description: str):
    """Generate and play a one-shot sound effect."""
    cache_key = _cache_key("sfx", description)
    cached = _get_cached(cache_key)

    if cached:
        print(f"  [SFX] Playing cached: {description}")
    else:
        print(f"  [SFX] Generating: {description}")
        try:
            sfx_iter = client.text_to_sound_effects.convert(
                text=description,
                duration_seconds=3.0,
                prompt_influence=0.5,
            )
            sfx_bytes = b"".join(sfx_iter)
            cached = _save_to_cache(cache_key, sfx_bytes)
            print(f"  [SFX] Generated ({len(sfx_bytes) // 1024}KB)")
        except Exception as e:
            print(f"  [SFX] Error: {e}")
            return

    try:
        sound = pygame.mixer.Sound(str(cached))
        CH_EFFECT.play(sound)
    except Exception as e:
        print(f"  [SFX] Playback error: {e}")


# ---------------------------------------------------------------------------
# Client tool handlers
# ---------------------------------------------------------------------------

def handle_set_scene(params: dict) -> str:
    """Called by the agent to set background music and ambient sounds."""
    mood = params.get("mood", "mysterious")
    environment = params.get("environment", "fantasy landscape")
    print(f"\n>> set_scene: mood={mood}, environment={environment}")
    thread = threading.Thread(
        target=_generate_and_play_music,
        args=(mood, environment),
        daemon=True,
    )
    thread.start()
    return f"Scene set: {mood} mood in {environment}. Music and ambiance loading."


def handle_play_effect(params: dict) -> str:
    """Called by the agent to play a one-shot sound effect."""
    description = params.get("description", "magical sound")
    print(f"\n>> play_effect: {description}")
    thread = threading.Thread(
        target=_generate_and_play_effect,
        args=(description,),
        daemon=True,
    )
    thread.start()
    return f"Playing sound effect: {description}"


# ---------------------------------------------------------------------------
# Agent creation / loading
# ---------------------------------------------------------------------------

def get_or_create_agent() -> str:
    """Load agent_id from config, or create a new Dungeon Master agent."""
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
        agent_id = config.get("agent_id")
        if agent_id:
            # Verify the agent still exists
            try:
                client.conversational_ai.agents.get(agent_id)
                print(f"Using existing agent: {agent_id}")
                return agent_id
            except Exception:
                print("Saved agent not found, creating new one...")

    print("Creating Dungeon Master agent...")
    response = client.conversational_ai.agents.create(
        name="Dungeon Master - Audio Adventure",
        conversation_config=ConversationalConfig(
            agent=AgentConfig(
                first_message=(
                    "Welcome, brave adventurer! I am your Dungeon Master. "
                    "A world of mystery and danger awaits you. Are you ready "
                    "to begin your quest?"
                ),
                language="en",
                prompt=PromptAgentApiModelOutput(
                    prompt=SYSTEM_PROMPT,
                    llm="gpt-4o-mini",
                    temperature=0.8,
                    max_tokens=300,
                    tools=[
                        {
                            "type": "client",
                            "name": "set_scene",
                            "description": (
                                "Set the background music mood and ambient environment "
                                "sounds. Call when the scene changes location or mood. "
                                "Parameters: mood (string, e.g. 'mysterious', 'epic', "
                                "'peaceful'), environment (string, e.g. 'deep cave', "
                                "'medieval tavern')."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "mood": {
                                        "type": "string",
                                        "description": "The mood of the music (e.g. mysterious, epic, dark, peaceful, tense)",
                                    },
                                    "environment": {
                                        "type": "string",
                                        "description": "The environment for ambient sounds (e.g. deep cave with dripping water)",
                                    },
                                },
                                "required": ["mood", "environment"],
                            },
                            "expects_response": True,
                        },
                        {
                            "type": "client",
                            "name": "play_effect",
                            "description": (
                                "Play a one-shot sound effect for dramatic moments. "
                                "Call for combat, discoveries, spell casting, doors, etc. "
                                "Parameter: description (string describing the sound)."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "Short description of the sound effect to generate",
                                    },
                                },
                                "required": ["description"],
                            },
                            "expects_response": True,
                        },
                    ],
                ),
            ),
            tts=TtsConversationalConfigOutput(
                model_id="eleven_flash_v2_5",
                voice_id=VOICE_ID,
                stability=0.5,
                similarity_boost=0.75,
            ),
        ),
    )

    agent_id = response.agent_id
    CONFIG_PATH.write_text(json.dumps({"agent_id": agent_id}, indent=2))
    print(f"Agent created: {agent_id}")
    return agent_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IMMERSIVE AUDIO ADVENTURE ENGINE")
    print("  Speak into your mic to play. Press Enter to quit.")
    print("=" * 60)

    agent_id = get_or_create_agent()

    # Register client tools
    client_tools = ClientTools()
    client_tools.register("set_scene", handle_set_scene)
    client_tools.register("play_effect", handle_play_effect)

    # Callbacks
    def on_agent_response(text: str):
        print(f"\nDM: {text}")

    def on_user_transcript(text: str):
        print(f"\nYou: {text}")

    # Create conversation
    conversation = Conversation(
        client=client,
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=client_tools,
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
    )

    print("\nConnecting to Dungeon Master...")
    conversation.start_session()
    print("Connected! Speak to play your adventure.\n")

    try:
        input("Press Enter to quit...\n")
    except (KeyboardInterrupt, EOFError):
        pass

    print("\nEnding adventure...")
    conversation.end_session()
    conversation.wait_for_session_end()

    # Fade out audio
    pygame.mixer.fadeout(1000)
    pygame.time.wait(1000)
    pygame.mixer.quit()

    print("Farewell, adventurer!")


if __name__ == "__main__":
    main()
