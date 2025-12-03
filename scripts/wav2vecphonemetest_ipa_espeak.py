import os
import csv
from pathlib import Path

import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# New IPA model
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
# === PATHS RELATIVE TO THIS SCRIPT ===
script_dir   = os.path.dirname(os.path.abspath(__file__))   # .../pronuncise_research/scripts
project_root = os.path.dirname(script_dir)                  # .../pronuncise_research

audio_dir = os.path.join(project_root, "audios")            # input audio folder
out_dir   = os.path.join(project_root, "data")              # where CSV will go
os.makedirs(out_dir, exist_ok=True)

out_csv = os.path.join(out_dir, "wav2vec_espeak_IPA_phonemes.csv")

# Accepted audio extensions
AUDIO_EXTS = {".wav"}

# -----------------------------------------------------------------------------
# LOAD MODEL + PROCESSOR
# -----------------------------------------------------------------------------

print(f"[INFO] Loading model: {MODEL_ID}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

target_sr = processor.feature_extractor.sampling_rate
print(f"[INFO] Using device: {device}, target SR = {target_sr} Hz")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def load_audio(path: Path, sr: int) -> torch.Tensor:
    """Load audio file as mono float32 numpy array at desired sampling rate."""
    audio, file_sr = librosa.load(path.as_posix(), sr=sr, mono=True)
    return audio


def transcribe_phonemes(path: Path) -> str:
    """Run the IPA phoneme model on one file and return the decoded phoneme sequence."""
    audio = load_audio(path, target_sr)

    inputs = processor(
        audio,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,          # fine, batch size = 1
    )

    input_values   = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    # Wav2Vec2Processor has batch_decode
    phoneme_seq = processor.batch_decode(predicted_ids)[0]

    return phoneme_seq


def load_existing_ids(csv_path: str) -> set:
    """Load sample_ids from existing CSV so we don't recompute them."""
    if not os.path.exists(csv_path):
        return set()

    ids = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" in row:
                ids.add(row["id"])
    return ids


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_path}")

    # Collect all audio files
    audio_files = sorted(
        p for p in audio_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )

    if not audio_files:
        print(f"[WARN] No audio files found in {audio_path}")
        return

    print(f"[INFO] Found {len(audio_files)} audio files in {audio_path}")

    # Load existing CSV if present
    existing_ids = load_existing_ids(out_csv)
    if existing_ids:
        print(f"[INFO] Found existing CSV with {len(existing_ids)} processed IDs")

    # Open CSV in append mode; write header only if file is new
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8", newline="") as f_out:
        fieldnames = ["id", "filename", "rel_path", "phonemes", "error"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for wav_path in audio_files:
            sample_id = wav_path.stem  # e.g. "71030" from "71030.wav"

            if sample_id in existing_ids:
                print(f"[SKIP] {wav_path.name} (id={sample_id}) already in CSV")
                continue

            print(f"[PROCESS] {wav_path.name} -> ID {sample_id}")

            row = {
                "id": sample_id,
                "filename": wav_path.name,
                "rel_path": os.path.relpath(wav_path.as_posix(), project_root),
                "phonemes": "",
                "error": "",
            }

            try:
                phonemes = transcribe_phonemes(wav_path)
                row["phonemes"] = phonemes
                print(f"         Phonemes: {phonemes[:80]}{'...' if len(phonemes) > 80 else ''}")
            except Exception as e:
                row["error"] = repr(e)
                print(f"[ERROR] Failed to transcribe {wav_path.name}: {e}")

            writer.writerow(row)
            f_out.flush()  # flush after each row for safety

    print(f"[DONE] Written results to: {out_csv}")


if __name__ == "__main__":
    main()
