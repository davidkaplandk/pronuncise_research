import os
import csv
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
import transformers  # just so we can print the version

# === 1) PATHS ===
audio_dir = r"C:\PY\Internship\mcv-spontaneous-de-v1.0\sps-corpus-1.0-2025-09-05-de\audios"
out_csv   = r"C:\PY\Internship\wav2vec2_phonemes.csv"

model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# === 2) DEVICE + MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, transformers={transformers.__version__}")

# Only feature extractor + model (NO processor, NO tokenizer)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
model.eval()

# Build int -> label map (phoneme symbols)
id2label_raw = model.config.id2label  # usually {"0": "<pad>", "1": "a", ...}
id2label = {int(k): v for k, v in id2label_raw.items()}

# Assume CTC blank is id 0 (standard for wav2vec2 CTC models)
blank_id = 0

def greedy_ctc_decode(logits):
    """
    logits: [1, T, vocab_size]
    returns: string like 'd ɪ z ə ʃ p ʁ a x ə ...'
    """
    # [1, T] -> list[int]
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    tokens = []
    prev = None
    for i in pred_ids:
        # skip blank
        if i == blank_id:
            prev = None
            continue
        # collapse repeats
        if i == prev:
            continue
        prev = i
        # map id -> phoneme symbol
        token = id2label.get(i, "")
        if token:
            tokens.append(token)

    # join with space; you can change to '' if you prefer no spaces
    return " ".join(tokens)

# === 3) LOAD EXISTING CSV (INCREMENTAL MODE) ===
existing_ids = set()
write_header = True

if os.path.exists(out_csv):
    print(f"Found existing CSV: {out_csv} — loading IDs")
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "id" in reader.fieldnames:
            for row in reader:
                existing_ids.add(row["id"])
    write_header = False

print(f"Already have {len(existing_ids)} IDs in CSV")

# === 4) PROCESS FILES AND APPEND NEW ROWS ===
mode = "a" if not write_header else "w"

with open(out_csv, mode, encoding="utf-8", newline="") as f_out:
    fieldnames = ["id", "phonemes"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    for fname in sorted(os.listdir(audio_dir)):
        if not fname.lower().endswith(".mp3"):
            continue

        stem = os.path.splitext(fname)[0]

        # ID extraction (you already renamed files to digits)
        if stem.isdigit():
            utt_id = stem
        elif "de-" in stem:
            utt_id = stem.split("de-")[-1]
        else:
            utt_id = stem

        if utt_id in existing_ids:
            print(f"[SKIP] {fname} (ID {utt_id}) already in CSV")
            continue

        path = os.path.join(audio_dir, fname)
        print(f"[PROCESS] {fname} -> ID {utt_id}")

        # --- load + resample to 16 kHz ---
        try:
            speech, sr = librosa.load(path, sr=16000)  # mono, 16 kHz
        except Exception as e:
            print(f"  !! Failed to load {fname}: {e}")
            continue

        # --- feature extraction + model ---
        try:
            with torch.no_grad():
                inputs = feature_extractor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                input_values = inputs["input_values"].to(device)
                logits = model(input_values).logits.cpu()
                phoneme_str = greedy_ctc_decode(logits)
        except Exception as e:
            print(f"  !! Failed to transcribe {fname}: {e}")
            continue

        phoneme_str = phoneme_str.strip()

        writer.writerow({"id": utt_id, "phonemes": phoneme_str})
        f_out.flush()
        existing_ids.add(utt_id)

print("Done.")
