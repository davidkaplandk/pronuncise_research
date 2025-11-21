import os
import csv
import torch
import whisperx

# fix ffmpeg not found 
os.environ["PATH"] += r";C:\ffmpeg"

# relative paths 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Folder with mp3s from common voice audio: ./audios
audio_dir = os.path.join(script_dir, "audios")

# Folder for CSV: ./test_sets
out_dir = os.path.join(script_dir, "test_sets")
os.makedirs(out_dir, exist_ok=True)

# One CSV with all transcripts
out_csv = os.path.join(out_dir, "transcription_whisperx_test.csv")

#whisperx model setup 
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # reduce if needed
compute_type = "float16" if device == "cuda" else "float32"

print(f"Using device: {device}, compute_type: {compute_type}")

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# loop over all mp3s and create csv

rows = []  # will store (id, transcription)

for fname in os.listdir(audio_dir):
    if not fname.lower().endswith(".mp3"):
        continue  # skip non-mp3 files

    audio_path = os.path.join(audio_dir, fname)

    # Example: spontaneous-speech-de-71253.mp3
    stem = os.path.splitext(fname)[0]  # spontaneous-speech-de-71253

    # Take ID in string after "de-"
    if "de-" in stem:
        id_part = stem.split("de-")[-1]  # -> "71253"
    else:
        id_part = stem  # fallback: full stem

    print(f"[TRANSCRIBE] {fname} (ID: {id_part})")

    # Transcribe
    result = model.transcribe(audio_path, batch_size=batch_size)

    # Combine segment texts into one transcript string
    full_text = " ".join(seg["text"].strip() for seg in result["segments"])

    # Save as one row for CSV
    rows.append((id_part, full_text))

# save the csv 

print(f"Writing CSV to: {out_csv}")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "transcription"])  # header
    writer.writerows(rows)

print("Done.")
