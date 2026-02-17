# Immersive Audio Adventure Engine

Voice-controlled text adventure using ElevenLabs Conversation SDK for real-time mic/voice interaction, with dynamically generated background music (Music API) and sound effects (SFX API) played via pygame.mixer.

## Usage

```bash
cd playground/audio-adventure
python adventure.py
```

First run creates the Dungeon Master agent via API. Subsequent runs reuse it.

## Sample Transcript (with working audio)

Music, ambient sounds, and sound effects all generate and play via pygame.mixer channels.

```
============================================================
  IMMERSIVE AUDIO ADVENTURE ENGINE
  Speak into your mic to play. Press Enter to quit.
============================================================
Using existing agent: agent_1701khn918vjfv4vvsjmfz1syj75

Connecting to Dungeon Master...
Connected! Speak to play your adventure.

DM: Welcome, brave adventurer! I am your Dungeon Master. A world of mystery
    and danger awaits you. Are you ready to begin your quest?

You: Yes, I am.

>> set_scene: mood=mysterious, environment=dense forest with rustling leaves
   and distant animal calls
  [Music] Generating: mysterious / dense forest with rustling leaves...
  [Music] Generated (468KB)
  [Ambient] Generating: dense forest with rustling leaves and distant animal calls
  [Ambient] Generated (157KB)

DM: You find yourself at the edge of an ancient, dense forest. The rustling
    of leaves whispers secrets of old, while distant animal calls echo through
    the trees. A narrow path stretches ahead, barely visible under the dappled
    sunlight. Will you venture deeper into the forest, or perhaps inspect the
    strange stone structure to your left?

You: Deeper into the forest

>> set_scene: mood=tense, environment=deep forest with thick underbrush and
   eerie silence
  [Music] Generating: tense / deep forest with thick underbrush...
  [Music] Generated (468KB)
  [Ambient] Generating: deep forest with thick underbrush and eerie silence
  [Ambient] Generated (157KB)

>> play_effect: soft glow illuminating the dark forest
  [SFX] Generating: soft glow illuminating the dark forest
  [SFX] Generated (47KB)

DM: You step cautiously into the depths of the forest. The trees grow thicker,
    and an eerie silence envelops the area, broken only by your footsteps
    crunching on fallen leaves. Suddenly, you notice a faint glow emanating
    from behind a cluster of trees. Do you choose to investigate the source
    of the light, or continue along the path?

You: Investigate source of light

DM: You move towards the soft glow, each step revealing more of the enchanted
    light. As you approach, you discover a shimmering orb hovering above a
    mossy stone altar. The orb pulses gently with a warm, inviting light. Do
    you reach out to touch the orb, or step back to observe it from a distance?

You: Reach out to touch the orb

>> play_effect: magical hum as your hand approaches the orb
  [SFX] Generating: magical hum as your hand approaches the orb
  [SFX] Generated (47KB)

DM: As your fingers draw near the orb, a gentle hum fills the air, and a
    tingling sensation courses through your hand. The orb feels warm and
    alive, resonating with a mysterious energy. Suddenly, the forest around
    you begins to shift and change. Do you hold onto the orb to see where
    it takes you, or pull your hand away?

You: Hold on.

>> set_scene: mood=otherworldly, environment=mystical realm with swirling
   colors and floating islands
  [Music] Generating: otherworldly / mystical realm with swirling colors...
  [Music] Generated (468KB)
  [Ambient] Generating: mystical realm with swirling colors and floating islands
  [Ambient] Generated (157KB)

DM: As you hold onto the orb, the forest around you dissolves, replaced by a
    mystical realm of swirling colors and floating islands. The air is filled
    with a symphony of ethereal sounds, and the ground beneath you feels
    light and buoyant. Ahead, a winding bridge of light connects the islands.
    Will you cross the bridge to explore further, or stay and examine the
    orb's magic in more detail?
```

## Audio Recording

`audio_adventure_soundtrack.m4a` contains the extracted soundtrack from a test session. Note: the human player's voice was not captured in the recording for unknown reasons — only the Dungeon Master's voice, background music, ambient sounds, and sound effects are audible.
