from allosaurus.app import read_recognizer
import os
import csv

# 1) Load model
model = read_recognizer()  # uses 'latest' universal model by default

# 2) Folder with your WAVs
samples_dir = r"audios"

# 3) List to store results
rows = []  # each entry will be (id, transcription)

# 4) Loop over all files
for sample in os.listdir(samples_dir):
    if not sample.lower().endswith(".wav"):
        continue

    # Full path to the audio file
    file_path = os.path.join(samples_dir, sample)

    # Extract ID from filename, e.g. "71030.wav" -> "71030"
    sample_id, _ = os.path.splitext(sample)

    # Run Allosaurus with German phone inventory
    try:
        transcription = model.recognize(file_path, "ipa")
    except Exception as e:
        print(f"Failed on {sample}: {e}")
        continue

    # Add to list
    rows.append((sample_id, transcription))

# 5) Save to CSV
out_csv = os.path.join(samples_dir, "allosaurus_transcriptions_3.csv")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "transcription"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to: {out_csv}")
