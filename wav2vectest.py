import os
import csv
import torch
from transformers import pipeline

# === 1) PATHS ===
# Folder with MP3s
audio_dir = r"C:\PY\Internship\mcv-spontaneous-de-v1.0\sps-corpus-1.0-2025-09-05-de\audios"

# Output folder for the CSV
base_out_dir = r"C:\PY\Internship"
os.makedirs(base_out_dir, exist_ok=True)

# One CSV with all transcripts
out_csv = os.path.join(base_out_dir, "wav2vec2_transcriptions.csv")

# === 2) Wav2Vec2 ASR PIPELINE SETUP ===
# device: 0 = first GPU, -1 = CPU
device = 0 if torch.cuda.is_available() else -1

print("Using device:", "cuda" if device == 0 else "cpu")

asr = pipeline(
    "automatic-speech-recognition",
    model="jonatasgrosman/wav2vec2-large-xlsr-53-german",  # German Wav2Vec2
    chunk_length_s=30,  # safe for somewhat longer files
    device=device
)

# === 3) LOOP OVER ALL MP3 FILES AND COLLECT ROWS ===
rows = []  # will store (id, transcription)

for fname in os.listdir(audio_dir):
    if not fname.lower().endswith(".mp3"):
        continue  # skip non-mp3 files

    audio_path = os.path.join(audio_dir, fname)

    # Example: spontaneous-speech-de-71253.mp3
    stem = os.path.splitext(fname)[0]  # spontaneous-speech-de-71253

    # Take ID in string after "de-"
    if "de-" in stem:
        id_part = stem.split("de-")[-1]  # -> eg "71253"
    else:
        id_part = stem  # fallback: full stem

    print(f"[TRANSCRIBE] {fname} (ID: {id_part})")

    # 3a) Transcribe with Wav2Vec2 pipeline
    result = asr(audio_path)

    # result is a dict like {"text": "...", "chunks": [...]} (chunks only if chunking)
    full_text = result["text"].strip()

    # Save as one row for CSV
    rows.append((id_part, full_text))

# === 4) WRITE CSV ===
print(f"Writing CSV to: {out_csv}")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "transcription"])  # header
    writer.writerows(rows)

print("Done.")
