import os
import csv
from pathlib import Path

import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Whisper IPA model (multilingual, incl. German)
MODEL_ID = "neurlang/ipa-whisper-base"

# Chunk length in seconds (25s is a safe value for Whisper)
CHUNK_DURATION_S = 25.0

# === PATHS RELATIVE TO THIS SCRIPT ===
script_dir   = os.path.dirname(os.path.abspath(__file__))   # .../pronuncise_research/scripts
project_root = os.path.dirname(script_dir)                  # .../pronuncise_research

audio_dir = os.path.join(project_root, "audios")            # input audio folder
out_dir   = os.path.join(project_root, "data")              # where CSV will go
os.makedirs(out_dir, exist_ok=True)

out_csv = os.path.join(out_dir, "whisper_phonemes_chunks.csv")

# Accepted audio extensions
AUDIO_EXTS = {".wav"}

# -----------------------------------------------------------------------------
# LOAD MODEL + PROCESSOR
# -----------------------------------------------------------------------------

print(f"[INFO] Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)

# --- Recommended IPA Whisper setup (adapted from model card) ---
# 1) Don't force language/task via context tokens in config
model.config.forced_decoder_ids = None

# 2) Allow all tokens (important for IPA symbols)
model.config.suppress_tokens = []
model.generation_config.suppress_tokens = []

# 3) Make generation follow the model config
model.generation_config.forced_decoder_ids = None
model.generation_config._from_model_config = True
# ---------------------------------------------------------------

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


def transcribe_chunk(chunk_audio: torch.Tensor) -> str:
    """Transcribe a single audio chunk (already at target_sr) to IPA."""
    inputs = processor(
        chunk_audio,
        sampling_rate=target_sr,
        return_tensors="pt"
    )

    input_features = inputs["input_features"].to(device)

    # Attention mask: (batch_size, time_steps) -> time_steps = last dim
    attn_mask = torch.ones(
        input_features.shape[0],   # batch
        input_features.shape[-1],  # time steps
        dtype=torch.long,
        device=device,
    )

    # max_new_tokens + prompt_length(~4) must be <= 448 for whisper-base
    gen_kwargs = dict(
        max_new_tokens=384,       # safe upper bound
        do_sample=False,
        no_repeat_ngram_size=3,
        language="german",        # force German decoding
        task="transcribe",        # ASR (not translation)
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            attention_mask=attn_mask,
            **gen_kwargs,
        )

    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    else:
        decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded[0].strip()


def transcribe_phonemes(path: Path) -> str:
    """
    Run the Whisper-IPA model on one file using chunking and return
    the concatenated decoded phoneme sequence.
    """
    audio = load_audio(path, target_sr)
    num_samples = len(audio)

    chunk_size = int(CHUNK_DURATION_S * target_sr)

    if num_samples <= chunk_size:
        # Short file -> no need to chunk
        return transcribe_chunk(audio)

    phoneme_chunks = []
    start = 0

    while start < num_samples:
        end = min(start + chunk_size, num_samples)
        chunk_audio = audio[start:end]

        # Skip empty chunks just in case
        if len(chunk_audio) == 0:
            break

        chunk_text = transcribe_chunk(chunk_audio)
        if chunk_text:
            phoneme_chunks.append(chunk_text)

        start = end  # no overlap; add overlap here if you want

    # Simple concatenation with a space between chunks
    return " ".join(phoneme_chunks).strip()


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
            sample_id = wav_path.stem  # e.g. "71334" from "71334.wav"

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
