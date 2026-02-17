"""
Workflow 19: Voice Remix and Character Migration Engine
=======================================================
Demonstrates the full voice migration pipeline:
  1. Generate a multi-speaker test audio (two voices having a conversation)
  2. Analyze source audio with STT + diarization to identify speakers
  3. Remix selected speaker's voice to a new voice identity
  4. Verify timing integrity between original and remixed audio
  5. Reassemble final audio with the remixed voice in place

Usage:
  python voice_migration.py              # Full pipeline with generated test audio
  python voice_migration.py <audio.mp3>  # Use your own multi-speaker audio file
"""

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment

# Load API key
load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Voice catalog (well-known ElevenLabs voice IDs) ---
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",   # Rachel - calm, female
    "adam": "pNInz6obpgDQGcFmaJgB",      # Adam - deep, male
    "josh": "TxGEqnHWrfWFTfGW9XjX",      # Josh - young, male
    "bella": "EXAVITQu4vr4xnSDxMaL",     # Bella - soft, female
    "george": "JBFqnCBsd6RMkjVDRZzb",    # George - British, male
}


@dataclass
class CharacterSegment:
    character: str
    text: str
    start_ms: int
    end_ms: int
    audio: AudioSegment


# ====================================================================
# Step 0: Generate a test audio file with two speakers
# ====================================================================

def generate_test_audio() -> Path:
    """Generate a two-speaker conversation using TTS for testing."""
    print("\n[Step 0] Generating test audio with two speakers...")

    lines = [
        ("rachel", "Good morning! I wanted to discuss the quarterly results with you."),
        ("adam", "Of course. The numbers look quite promising this quarter."),
        ("rachel", "Revenue was up fifteen percent compared to last year."),
        ("adam", "That's great news. And our customer retention rate improved significantly."),
        ("rachel", "Exactly. I think the new product launch made a real difference."),
        ("adam", "I agree completely. Shall we prepare the board presentation together?"),
    ]

    combined = AudioSegment.silent(duration=300)  # small leading silence

    for i, (speaker, text) in enumerate(lines):
        voice_id = VOICES[speaker]
        print(f"  Generating line {i+1}/{len(lines)}: [{speaker}] {text[:50]}...")

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.6,
                similarity_boost=0.75,
                style=0.0,
            ),
        )

        audio_bytes = b"".join(chunk for chunk in audio_stream)
        segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

        combined += segment
        combined += AudioSegment.silent(duration=400)  # pause between lines

    output_path = OUTPUT_DIR / "test_conversation.mp3"
    combined.export(str(output_path), format="mp3")
    duration_s = len(combined) / 1000
    print(f"  Test audio generated: {output_path} ({duration_s:.1f}s)")
    return output_path


# ====================================================================
# Step 1: Analyze source audio (STT + diarization)
# ====================================================================

def analyze_source_audio(audio_path: str | Path) -> list[CharacterSegment]:
    """Transcribe with diarization to identify character segments."""
    print(f"\n[Step 1] Analyzing source audio: {audio_path}")

    with open(audio_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v2",
            diarize=True,
            timestamps_granularity="word",
        )

    source_audio = AudioSegment.from_file(str(audio_path))

    # Group consecutive words by speaker into segments
    segments: list[CharacterSegment] = []
    current_speaker = None
    current_words = []
    current_start_ms = 0

    for word in result.words:
        speaker = str(getattr(word, "speaker_id", None) or "unknown")

        if speaker != current_speaker and current_words:
            end_ms = int(current_words[-1].end * 1000)
            text = " ".join(w.text for w in current_words if w.type == "word")
            if text.strip():
                segments.append(CharacterSegment(
                    character=current_speaker,
                    text=text,
                    start_ms=current_start_ms,
                    end_ms=end_ms,
                    audio=source_audio[current_start_ms:end_ms],
                ))
            current_words = []
            current_start_ms = int(word.start * 1000)

        current_speaker = speaker
        current_words.append(word)

    # Final segment
    if current_words:
        end_ms = int(current_words[-1].end * 1000)
        text = " ".join(w.text for w in current_words if w.type == "word")
        if text.strip():
            segments.append(CharacterSegment(
                character=current_speaker,
                text=text,
                start_ms=current_start_ms,
                end_ms=end_ms,
                audio=source_audio[current_start_ms:end_ms],
            ))

    # Summary
    speakers = sorted(set(s.character for s in segments))
    print(f"  Found {len(segments)} segments across {len(speakers)} speakers:")
    for spk in speakers:
        spk_segs = [s for s in segments if s.character == spk]
        total_ms = sum(s.end_ms - s.start_ms for s in spk_segs)
        print(f"    {spk}: {len(spk_segs)} segments, {total_ms/1000:.1f}s total")
        for s in spk_segs:
            print(f"      [{s.start_ms/1000:.1f}s-{s.end_ms/1000:.1f}s] \"{s.text[:60]}\"")

    return segments


# ====================================================================
# Step 2: Voice Remix (change voice identity, preserve delivery)
# ====================================================================

def remix_voice(segment: CharacterSegment, target_voice_id: str) -> AudioSegment:
    """Transform the voice while preserving the original delivery timing."""
    buffer = io.BytesIO()
    segment.audio.export(buffer, format="mp3")
    audio_bytes = buffer.getvalue()

    # The speech-to-speech / voice changer API preserves prosody and timing
    remixed_stream = client.speech_to_speech.convert(
        voice_id=target_voice_id,
        audio=audio_bytes,
        model_id="eleven_english_sts_v2",
        output_format="mp3_44100_128",
    )

    remixed_bytes = b"".join(chunk for chunk in remixed_stream)
    return AudioSegment.from_mp3(io.BytesIO(remixed_bytes))


# ====================================================================
# Step 3: Verify timing integrity
# ====================================================================

def verify_timing(original: CharacterSegment, remixed: AudioSegment) -> dict:
    """Compare durations to check timing drift."""
    orig_ms = len(original.audio)
    new_ms = len(remixed)
    drift_ms = abs(new_ms - orig_ms)
    drift_pct = (drift_ms / orig_ms * 100) if orig_ms > 0 else 0

    return {
        "original_ms": orig_ms,
        "remixed_ms": new_ms,
        "drift_ms": drift_ms,
        "drift_pct": round(drift_pct, 1),
        "acceptable": drift_pct < 15.0,  # <15% drift for STS is reasonable
    }


# ====================================================================
# Step 4: Reassemble audio
# ====================================================================

def reassemble(
    original_audio_path: str | Path,
    segments: list[CharacterSegment],
    remixed_segments: dict[int, AudioSegment],
) -> AudioSegment:
    """Rebuild audio with remixed segments placed at their original timestamps."""
    original = AudioSegment.from_file(str(original_audio_path))
    output = AudioSegment.silent(duration=len(original))

    for i, segment in enumerate(segments):
        if i in remixed_segments:
            new_audio = remixed_segments[i]
            target_ms = segment.end_ms - segment.start_ms
            # Trim or pad to match original slot
            if len(new_audio) > target_ms:
                new_audio = new_audio[:target_ms]
            elif len(new_audio) < target_ms:
                new_audio = new_audio + AudioSegment.silent(duration=target_ms - len(new_audio))
            output = output.overlay(new_audio, position=segment.start_ms)
        else:
            output = output.overlay(segment.audio, position=segment.start_ms)

    return output


# ====================================================================
# Full pipeline
# ====================================================================

def migrate_voices(
    audio_path: str | Path,
    speaker_voice_map: dict[str, str],
    output_filename: str = "migrated_output.mp3",
) -> dict:
    """
    Full voice migration pipeline.

    Args:
        audio_path: Path to source audio file
        speaker_voice_map: e.g. {"speaker_0": "JBFqnCBsd6RMkjVDRZzb"}
        output_filename: Name for the output file
    """
    print("=" * 60)
    print("VOICE MIGRATION ENGINE")
    print("=" * 60)

    # Analyze
    segments = analyze_source_audio(audio_path)

    # Find which speakers to remap
    found_speakers = sorted(set(s.character for s in segments))
    remap_speakers = {k: v for k, v in speaker_voice_map.items() if k in found_speakers}

    if not remap_speakers:
        print(f"\n  No matching speakers to remap.")
        print(f"  Available speakers: {found_speakers}")
        print(f"  Requested: {list(speaker_voice_map.keys())}")
        return {"status": "no_match", "speakers": found_speakers}

    print(f"\n[Step 2] Remixing voices...")
    print(f"  Remapping: {remap_speakers}")

    remixed: dict[int, AudioSegment] = {}
    timing_reports = []

    for i, segment in enumerate(segments):
        if segment.character in remap_speakers:
            target_voice = remap_speakers[segment.character]
            print(f"\n  Segment {i}: {segment.character}")
            print(f"    Text: \"{segment.text[:70]}\"")
            print(f"    Duration: {len(segment.audio)}ms")
            print(f"    Target voice: {target_voice}")

            new_audio = remix_voice(segment, target_voice)
            remixed[i] = new_audio

            timing = verify_timing(segment, new_audio)
            timing_reports.append({"segment": i, "character": segment.character, **timing})
            status = "OK" if timing["acceptable"] else "DRIFT WARNING"
            print(f"    Timing: {timing['original_ms']}ms -> {timing['remixed_ms']}ms "
                  f"({timing['drift_pct']}% drift) [{status}]")

    # Reassemble
    print(f"\n[Step 3] Reassembling audio ({len(remixed)}/{len(segments)} segments remixed)...")
    output = reassemble(audio_path, segments, remixed)

    # Export
    output_path = OUTPUT_DIR / output_filename
    output.export(str(output_path), format="mp3")
    print(f"\n[Step 4] Exported: {output_path} ({len(output)/1000:.1f}s)")

    # Save report
    report = {
        "source": str(audio_path),
        "output": str(output_path),
        "speakers_found": found_speakers,
        "speakers_remixed": list(remap_speakers.keys()),
        "segments_total": len(segments),
        "segments_remixed": len(remixed),
        "timing_reports": timing_reports,
    }
    report_path = OUTPUT_DIR / "migration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("MIGRATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Original: {audio_path}")
    print(f"  Migrated: {output_path}")
    print(f"  Segments remixed: {len(remixed)}/{len(segments)}")
    acceptable = sum(1 for t in timing_reports if t["acceptable"])
    print(f"  Timing checks: {acceptable}/{len(timing_reports)} within tolerance")

    return report


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    # If an audio file is provided, use it; otherwise generate test audio
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            print(f"Error: {audio_path} not found")
            sys.exit(1)
    else:
        audio_path = generate_test_audio()

    # Run migration: remap speaker_0 (the first detected speaker) to George's voice
    # You can change this after seeing which speakers are detected
    result = migrate_voices(
        audio_path=audio_path,
        speaker_voice_map={
            "speaker_0": VOICES["george"],  # Remap first speaker to George (British male)
        },
        output_filename="migrated_conversation.mp3",
    )
