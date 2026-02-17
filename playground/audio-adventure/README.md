# Immersive Audio Adventure Engine

Voice-controlled text adventure using ElevenLabs Conversation SDK for real-time mic/voice interaction, with dynamically generated background music (Music API) and sound effects (SFX API) played via pygame.mixer.

## Usage

```bash
cd playground/audio-adventure
python adventure.py
```

First run creates the Dungeon Master agent via API. Subsequent runs reuse it.

## Known Issues

- **Sound effects not triggered**: The agent reads tool calls as text (e.g. `set_scene({ mood: "mysterious", ... })`) instead of executing them as client tools. The `set_scene` and `play_effect` handlers are registered but the agent outputs tool invocations as narrated speech rather than actual tool calls. This means background music and sound effects are not generated/played.

## Sample Transcript

```
============================================================
  IMMERSIVE AUDIO ADVENTURE ENGINE
  Speak into your mic to play. Press Enter to quit.
============================================================
Using existing agent: agent_8201khn80xm3f89rmaf32gyh2few

Connecting to Dungeon Master...
Connected! Speak to play your adventure.

DM: Welcome, brave adventurer! I am your Dungeon Master. A world of mystery
    and danger awaits you. Are you ready to begin your quest?

You: Yes, I am.

DM: set_scene({ mood: "mysterious", environment: "dense forest with whispering
    leaves" });

    You find yourself standing in a dense forest, where the moonlight barely
    pierces through the thick canopy. The air is filled with the sounds of
    nocturnal creatures, and the faint rustling of leaves hints at unseen
    movement. A narrow path winds deeper into the woods, while to your left,
    you hear the gentle flow of a stream.

    What would you like to do? You can choose to follow the path deeper into
    the forest, investigate the stream, or proceed cautiously off the trail.

You: Deeper into the forest.

DM: play_effect({ description: "soft footsteps crunching on leaves" });

    You step onto the narrow path, your footsteps muffled by the carpet of
    leaves beneath you. As you venture deeper into the forest, the shadows
    grow thicker, and the sounds of the night transform into an eerie silence.
    Suddenly, you come upon a clearing where an ancient stone altar sits,
    covered in moss and surrounded by flickering blue flames.

    What will you do next? You can examine the altar, search the area for
    clues, or retreat back to the path.

You: Search the area for clues.

DM: play_effect({ description: "soft rustling as leaves are moved" });

    You carefully search the area around the ancient stone altar, moving aside
    the overgrown vines and brushing away the leaves. Among the stones, you
    discover a small, intricately carved dagger, its blade shimmering even in
    the low light. You also find strange symbols etched into the ground,
    hinting at a forgotten ritual.

    What would you like to do now? You can take the dagger, examine the
    symbols more closely, or inspect the altar itself.

You: Take the dagger.

DM: play_effect({ description: "metallic clink as the dagger is picked up" });

    You grasp the dagger firmly, feeling its cool metal in your hand. As you
    lift it from the ground, a rush of energy courses through you, and you
    sense that this blade has some magical properties. With your new weapon in
    your possession, the night feels a little less foreboding.

    What will you do next? You can examine the symbols on the ground, inspect
    the altar, or continue deeper into the forest.

You: Continue deeper into the forest.

DM: set_scene({ mood: "tense", environment: "dark forest with flickering
    shadows" });

    With the dagger in hand, you press onward into the depths of the forest.
    The air grows heavier, and the shadows seem to twist and dance around you.
    Suddenly, you hear a low growl echoing through the trees, sending a chill
    down your spine. A pair of glowing eyes appear in the darkness, watching
    you intently.

    What will you do? You can confront the creature, attempt to sneak past
    it, or find a place to hide.

You: Confront the creature.

DM: play_effect({ description: "a tense growl reverberating through the
    forest" });

    You stand your ground, gripping the dagger tightly as you confront the
    creature. Emerging from the shadows, it reveals itself to be a large
    wolf, its fur dark as night and eyes gleaming like emeralds. It sniffs
    the air, sizing you up, and you sense an intelligence behind its gaze.

    What will you do next? You can attempt to communicate with the wolf,
    prepare to defend yourself, or try to intimidate it with a show of
    strength.
```
