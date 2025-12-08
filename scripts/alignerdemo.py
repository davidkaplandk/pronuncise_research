"""
End-to-end pipeline:

1. (Optional) Run Montreal Forced Aligner (MFA) on a German corpus.
2. Parse TextGrids produced by MFA to get word & phone intervals.
3. Load original transcripts and tokenize into words.
4. Generate canonical IPA for each word with phonemizer + eSpeak-NG (German).
5. Match text words, MFA word intervals, MFA phones, and canonical IPA.
6. Write a CSV with one row per word: timings + orthography + IPA.

Prereqs (pip):

    pip install textgrid phonemizer

System:

    - MFA installed and on PATH (`mfa` CLI).
    - eSpeak NG installed, and its library path known (for phonemizer backend=espeak).

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

# ============================================================================
# CONFIG
# ============================================================================

# --- Paths you must set ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # adjust if needed

CORPUS_DIR      = PROJECT_ROOT / "corpus_german"        # wav + txt
MFA_OUTPUT_DIR  = PROJECT_ROOT / "mfa_output"           # where TextGrids go
CSV_OUT_PATH    = PROJECT_ROOT / "data" / "aligned_words_ipa.csv"

GERMAN_DICT     = PROJECT_ROOT / "mfa_models" / "german.dict"   # MFA pronunciation dict
GERMAN_ACOUSTIC = PROJECT_ROOT / "mfa_models" / "german_model.zip"

# Path to eSpeak-NG DLL for Windows (adapt to your installation):
ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# Tier names in MFA TextGrid; adjust if yours are different
WORD_TIER_NAMES  = ["words", "word"]
PHONE_TIER_NAMES = ["phones", "phone"]

# If True, call MFA from this script; if you already have TextGrids, set False.
RUN_MFA_FROM_SCRIPT = False

# Small time tolerance when grouping phones into words
EPS = 1e-3


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
    raise ValueError(f"Could not find tier named one of {possible_names}, "
                     f"found tiers: {[t.name for t in tg.tiers]}")


def load_word_and_phone_intervals(tg_path: Path):
    """
    Load word and phone intervals from a TextGrid.

    Returns:
        word_intervals: List[(start, end, label)]
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
    Load transcript from CORPUS_DIR/utt_id.txt.
    Returns None if not found.
    """
    txt_path = CORPUS_DIR / f"{utt_id}.txt"
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding="utf-8").strip()


def simple_tokenize(text: str) -> List[str]:
    """
    Very simple whitespace tokenizer.
    You can replace this by a more robust tokenizer if needed.
    """
    return [w for w in text.split() if w]


def ipa_for_words(words: List[str]) -> List[str]:
    """
    Use phonemizer + eSpeak-NG (de) to obtain canonical IPA per word.
    Returns a list of the same length as `words`.
    """
    if not words:
        return []

    # Initialise eSpeak library path (needed on Windows)
    EspeakWrapper.set_library(ESPEAK_DLL)

    # Phonemizer expects a list of strings
    # We will phonemize the whole sequence at once (faster).
    separator = Separator(phone=" ", word=" ")

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

    # Prepare CSV
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
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # Process each TextGrid / utterance
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
                print(f"  [WARN] No transcript .txt for {utt_id}, skipping.")
                continue

            words_text = simple_tokenize(transcript)
            if not words_text:
                print(f"  [WARN] Empty transcript for {utt_id}, skipping.")
                continue

            # Generate canonical IPA for each word (same length as words_text)
            ipa_list = ipa_for_words(words_text)

            # Now we need to align words_text with word_intervals
            # Simplest assumption: same order, similar count.
            if len(words_text) != len(word_intervals):
                print(
                    f"  [WARN] Word count mismatch for {utt_id}: "
                    f"text has {len(words_text)}, TextGrid has {len(word_intervals)}"
                )
                # We'll align up to the min length and ignore the rest
            n = min(len(words_text), len(word_intervals), len(ipa_list))

            # Path to wav is typically in CORPUS_DIR
            wav_path = CORPUS_DIR / f"{utt_id}.wav"

            for i in range(n):
                w_start, w_end, grid_label = word_intervals[i]
                word_text = words_text[i]
                ipa_word  = ipa_list[i]

                # Gather MFA phones for this word
                mfa_ph_list = phones_for_word(w_start, w_end, phone_intervals, eps=EPS)
                mfa_ph_str  = " ".join(mfa_ph_list)

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
                }
                writer.writerow(row)

    print(f"[DONE] Wrote word-level IPA alignment to: {CSV_OUT_PATH}")


if __name__ == "__main__":
    main()
