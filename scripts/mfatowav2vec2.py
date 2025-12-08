"""
End-to-end pipeline:

1. (Optional) Run Montreal Forced Aligner (MFA) on a German corpus.
2. Parse TextGrids produced by MFA to get word & phone intervals.
3. Load original transcripts and tokenize into words.
4. Generate canonical IPA for each word with phonemizer + eSpeak-NG (German).
5. Match text words, MFA word intervals, MFA phones, and canonical IPA.
6. Additionally: Extract audio for each word and run a phonetic ASR model
   (e.g. wav2vec2-espeak) on that chunk.
7. Write a CSV with one row per word: timings + orthography + IPA + MFA phones + ASR phones.
"""

import os
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import textgrid
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper

### NEW: imports for ASR
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ============================================================================
# CONFIG
# ============================================================================

# Use pathlib consistently
script_dir   = Path(__file__).resolve().parent
PROJECT_ROOT = script_dir.parent

# Folders with your corpus (wav + txt/lab) and MFA output
CORPUS_DIR      = PROJECT_ROOT / "mfa_corpus_german"      # <utt_id>.wav + <utt_id>.lab/.txt
MFA_OUTPUT_DIR  = PROJECT_ROOT / "mfa_output"             # MFA TextGrids
CSV_OUT_PATH    = PROJECT_ROOT / "data" / "aligned_words_ipa.csv"

# MFA models
GERMAN_DICT     = Path(
    r"C:\Users\david\Documents\MFA\pretrained_models\dictionary\german_mfa.dict"
)
GERMAN_ACOUSTIC = Path(
    r"C:\Users\david\Documents\MFA\pretrained_models\acoustic\german_mfa.zip"
)

# Path to eSpeak-NG DLL for Windows
ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# Tier names in MFA TextGrid; adjust if yours are different
WORD_TIER_NAMES  = ["words", "word"]
PHONE_TIER_NAMES = ["phones", "phone"]

# If True, call MFA from this script; if you already have TextGrids, set False.
RUN_MFA_FROM_SCRIPT = False

# Small time tolerance when grouping phones into words
EPS = 1e-3

# Initialize eSpeak once
EspeakWrapper.set_library(ESPEAK_DLL)

### NEW: ASR model config
ASR_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

print(f"[ASR] Loading phonetic ASR model: {ASR_MODEL_ID}")
asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_ID)
asr_model     = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)
asr_model.eval()

TARGET_SR = asr_processor.feature_extractor.sampling_rate
print(f"[ASR] Using device: {device}, target SR = {TARGET_SR} Hz")

# ============================================================================
# MFA RUNNER
# ============================================================================

def run_mfa_align():
    """Run Montreal Forced Aligner on the corpus directory."""
    MFA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MFA] Running MFA align on {CORPUS_DIR} -> {MFA_OUTPUT_DIR}")

    cmd = [
        "mfa", "align",
        str(CORPUS_DIR),
        str(GERMAN_DICT),
        str(GERMAN_ACOUSTIC),
        str(MFA_OUTPUT_DIR),
        "--clean",          # optional: remove temp files
    ]
    print("[MFA] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[MFA] Done.")

# ============================================================================
# TEXTGRID HELPERS
# ============================================================================

def get_tier(tg: textgrid.TextGrid, possible_names: List[str]) -> textgrid.IntervalTier:
    """
    Get the first tier whose name matches any of `possible_names` (case-insensitive),
    otherwise raise an error.
    """
    names_lower = [n.lower() for n in possible_names]
    for tier in tg.tiers:
        if tier.name.lower() in names_lower:
            return tier
    raise ValueError(
        f"Could not find tier named one of {possible_names}, "
        f"found tiers: {[t.name for t in tg.tiers]}"
    )


def load_word_and_phone_intervals(tg_path: Path):
    """
    Load word and phone intervals from a TextGrid.

    Returns:
        word_intervals:  List[(start, end, label)]
        phone_intervals: List[(start, end, label)]
    """
    tg = textgrid.TextGrid.fromFile(str(tg_path))

    word_tier  = get_tier(tg, WORD_TIER_NAMES)
    phone_tier = get_tier(tg, PHONE_TIER_NAMES)

    word_intervals = [
        (iv.minTime, iv.maxTime, iv.mark)
        for iv in word_tier
        if iv.mark.strip()
    ]
    phone_intervals = [
        (iv.minTime, iv.maxTime, iv.mark)
        for iv in phone_tier
        if iv.mark.strip()
    ]
    return word_intervals, phone_intervals

# ============================================================================
# TEXT + IPA
# ============================================================================

def load_transcript_for_utt(utt_id: str) -> Optional[str]:
    """
    Load transcript from CORPUS_DIR/utt_id.lab or utt_id.txt.
    Returns None if not found.
    """
    lab_path = CORPUS_DIR / f"{utt_id}.lab"
    txt_path = CORPUS_DIR / f"{utt_id}.txt"

    if lab_path.exists():
        return lab_path.read_text(encoding="utf-8").strip()
    elif txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    else:
        return None


def simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return [w for w in text.split() if w]


def ipa_for_words(words: List[str]) -> List[str]:
    """
    Use phonemizer + eSpeak-NG (de) to obtain canonical IPA per word.
    Returns a list of the same length as `words`.
    """
    if not words:
        return []

    # Phones separated by spaces; no special word separator needed
    separator = Separator(phone=" ", word="")

    ipa_list = phonemize(
        words,
        language="de",
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        separator=separator,
        njobs=1,
    )
    return ipa_list

# ============================================================================
# ASR HELPERS (NEW)
# ============================================================================

def load_wave_for_utt(utt_id: str) -> Tuple[torch.Tensor, int, Path]:
    """
    Load wav for utterance as mono, resampled to TARGET_SR.
    Returns (waveform_1d, sr, wav_path).
    """
    wav_path = CORPUS_DIR / f"{utt_id}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found for {utt_id}: {wav_path}")

    waveform, sr = torchaudio.load(str(wav_path))  # [C, T]

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        sr = TARGET_SR

    waveform = waveform.squeeze(0)  # -> [T]
    return waveform, sr, wav_path


def asr_phones_for_segment(segment: torch.Tensor, sr: int) -> str:
    """
    Run wav2vec2 ASR on a 1D mono segment and return decoded phoneme string.
    """
    if segment.numel() == 0:
        return ""

    # HuggingFace processor accepts numpy arrays or lists
    audio_np = segment.numpy()

    inputs = asr_processor(
        audio_np,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = asr_model(input_values, attention_mask=attention_mask).logits

    pred_ids = logits.argmax(dim=-1)

    if hasattr(asr_processor, "batch_decode"):
        decoded = asr_processor.batch_decode(pred_ids)
    else:
        decoded = asr_processor.tokenizer.batch_decode(pred_ids)

    return decoded[0]

# ============================================================================
# ALIGNING PHONES TO WORDS
# ============================================================================

def phones_for_word(
    word_start: float,
    word_end: float,
    phone_intervals: List[Tuple[float, float, str]],
    eps: float = EPS
) -> List[str]:
    """
    Collect phones whose time span lies within [word_start - eps, word_end + eps].
    Returns list of phone labels (MFA phone set).
    """
    phones = []
    for ph_start, ph_end, ph_label in phone_intervals:
        if ph_start >= (word_start - eps) and ph_end <= (word_end + eps):
            phones.append(ph_label)
    return phones

# ============================================================================
# MAIN CSV GENERATION
# ============================================================================

def main():
    # Optionally run MFA
    if RUN_MFA_FROM_SCRIPT:
        run_mfa_align()
    else:
        print("[INFO] Skipping MFA run (RUN_MFA_FROM_SCRIPT = False)")

    if not MFA_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"MFA output dir not found: {MFA_OUTPUT_DIR}")

    MFA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    textgrids = sorted(MFA_OUTPUT_DIR.glob("*.TextGrid"))
    if not textgrids:
        print(f"[WARN] No TextGrid files found in {MFA_OUTPUT_DIR}")
        return

    print(f"[INFO] Found {len(textgrids)} TextGrids")

    with CSV_OUT_PATH.open("w", encoding="utf-8", newline="") as f_out:
        fieldnames = [
            "utt_id",
            "wav_path",
            "word_index",
            "word_text",
            "t_start",
            "t_end",
            "duration",
            "ipa_canonical",
            "mfa_phones",   # phones from MFA phone tier (space-separated)
            "asr_phones",   # NEW: phones from wav2vec2 segment decoding
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for tg_path in textgrids:
            utt_id = tg_path.stem  # "71030" from "71030.TextGrid"
            print(f"[UTT] {utt_id}")

            # Load MFA intervals
            try:
                word_intervals, phone_intervals = load_word_and_phone_intervals(tg_path)
            except Exception as e:
                print(f"  [ERROR] Failed to read {tg_path.name}: {e}")
                continue

            # Load transcript
            transcript = load_transcript_for_utt(utt_id)
            if transcript is None:
                print(f"  [WARN] No transcript .lab/.txt for {utt_id}, skipping.")
                continue

            words_text = simple_tokenize(transcript)
            if not words_text:
                print(f"  [WARN] Empty transcript for {utt_id}, skipping.")
                continue

            # Canonical IPA (same length as words_text)
            ipa_list = ipa_for_words(words_text)

            # Load audio once per utterance
            try:
                waveform, sr, wav_path = load_wave_for_utt(utt_id)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}, skipping ASR for this utt.")
                # Still write rows, but asr_phones will be empty
                waveform, sr, wav_path = None, None, CORPUS_DIR / f"{utt_id}.wav"

            # Align list positions: text words vs. TextGrid words
            if len(words_text) != len(word_intervals):
                print(
                    f"  [WARN] Word count mismatch for {utt_id}: "
                    f"text has {len(words_text)}, TextGrid has {len(word_intervals)}"
                )

            n = min(len(words_text), len(word_intervals), len(ipa_list))

            for i in range(n):
                w_start, w_end, grid_label = word_intervals[i]
                word_text = words_text[i]
                ipa_word  = ipa_list[i]

                # MFA phones
                mfa_ph_list = phones_for_word(w_start, w_end, phone_intervals, eps=EPS)
                mfa_ph_str  = " ".join(mfa_ph_list)

                # ASR phones for this word segment
                if waveform is not None and sr is not None:
                    start_sample = max(0, int(round(w_start * sr)))
                    end_sample   = min(waveform.shape[0], int(round(w_end * sr)))
                    if end_sample <= start_sample:
                        asr_ph_str = ""
                    else:
                        segment = waveform[start_sample:end_sample]
                        asr_ph_str = asr_phones_for_segment(segment, sr)
                else:
                    asr_ph_str = ""

                row = {
                    "utt_id": utt_id,
                    "wav_path": str(wav_path),
                    "word_index": i,
                    "word_text": word_text,
                    "t_start": f"{w_start:.3f}",
                    "t_end": f"{w_end:.3f}",
                    "duration": f"{(w_end - w_start):.3f}",
                    "ipa_canonical": ipa_word,
                    "mfa_phones": mfa_ph_str,
                    "asr_phones": asr_ph_str,
                }
                writer.writerow(row)

    print(f"[DONE] Wrote word-level IPA + MFA + ASR to: {CSV_OUT_PATH}")


if __name__ == "__main__":
    main()
