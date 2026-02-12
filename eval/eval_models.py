#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate two Wav2Vec2-CTC phoneme models on the same CSV/TextGrid test set.

Models:
  1) baseline (no word-boundary token):        hf_wav2vec2_de_phonemes
  2) with word-boundary token "|" (spaces):    hf_wav2vec2_de_phonemes_with_spaces

Input:
  test_wav.csv with columns: audio_file, grid_file

What it computes (valuable metrics):
  - PER (phoneme error rate): edit_distance(tokens) / #ref_tokens
  - SER (sequence error rate): % utterances with >=1 token error
  - Avg ref/hyp token length
  - For the "|" model additionally:
      * PER_no_delim (PER excluding "|" tokens)
      * Word-level WER by splitting on "|"
      * Boundary precision/recall/F1 using "|" positions (cumulative phone index)

It also writes per-utterance outputs to CSV for quick error analysis.

Usage (PowerShell):
  python eval_models.py

Optional:
  python eval_models.py --batch_size 4 --device cuda

Notes:
  - This script does NOT require `evaluate` or internet access.
  - It uses greedy CTC decoding (argmax + collapse repeats + remove blank).
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from praatio import textgrid as praatio_textgrid
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# -----------------
# Defaults (your paths)
# -----------------
DEFAULT_MODEL_A = Path(r"C:\PY\pronuncise_research\eval\eval_modelB_per_utt.csv")
DEFAULT_MODEL_B = Path(r"C:\PY\pronuncise_research\eval\eval_modelB_per_utt.csv")
DEFAULT_TESTCSV = Path(r"C:\PY\commonphone\CP\de\test_wav.csv")  ## change for your path (commonphone dataset)

# If your CSV stores relative paths, these are used as bases:
DEFAULT_MODEL_A = Path("eval_modelA_per_utt.csv")
DEFAULT_MODEL_B = Path("eval_modelB_per_utt.csv")

TARGET_SR = 16000
TIER_NAME = "KAN-MAU"
WORD_DELIM = "|"


# -----------------
# CSV utils
# -----------------
def load_two_col_paths(csv_path: Path, audio_dir: Path, grids_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    if "audio_file" not in df.columns or "grid_file" not in df.columns:
        raise ValueError(f"{csv_path}: needs columns audio_file + grid_file, got {list(df.columns)}")

    df = df[["audio_file", "grid_file"]].copy()

    def to_abs(p: str, base_dir: Path) -> str:
        pp = Path(str(p))
        if pp.exists():
            return str(pp.resolve())
        cand = base_dir / pp
        if cand.exists():
            return str(cand.resolve())
        cand2 = base_dir / pp.name
        if cand2.exists():
            return str(cand2.resolve())
        raise FileNotFoundError(f"Not found: '{p}' (looked in '{base_dir}')")

    df["audio_path"] = df["audio_file"].apply(lambda x: to_abs(x, audio_dir))
    df["grid_path"]  = df["grid_file"].apply(lambda x: to_abs(x, grids_dir))
    return df


# -----------------
# TextGrid -> tokens
# -----------------
def load_phones_flat(tg_path: Path, tier_name: str = TIER_NAME) -> List[str]:
    tg = praatio_textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if tier_name not in tg.tierNames:
        raise ValueError(f"{tg_path}: missing tier '{tier_name}'. Available: {tg.tierNames}")
    tier = tg.getTier(tier_name)

    phones: List[str] = []
    for (_, _, label) in tier.entries:
        lab = (label or "").strip()
        if lab == "":
            continue
        phones.extend(lab.split())
    if not phones:
        raise ValueError(f"{tg_path}: extracted 0 phones from tier {tier_name}")
    return phones


def load_phones_with_word_boundaries(tg_path: Path, tier_name: str = TIER_NAME, word_delim: str = WORD_DELIM) -> List[str]:
    """
    Treat each non-empty interval in tier as a word, whose label is a phone sequence.
    Flatten and insert word_delim between intervals/words.
    """
    tg = praatio_textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if tier_name not in tg.tierNames:
        raise ValueError(f"{tg_path}: missing tier '{tier_name}'. Available: {tg.tierNames}")
    tier = tg.getTier(tier_name)

    words_as_phone_lists: List[List[str]] = []
    for (_, _, label) in tier.entries:
        lab = (label or "").strip()
        if lab == "":
            continue
        phones = lab.split()
        if phones:
            words_as_phone_lists.append(phones)

    if not words_as_phone_lists:
        raise ValueError(f"{tg_path}: extracted 0 word phone-groups from tier {tier_name}")

    out: List[str] = []
    for wi, phones in enumerate(words_as_phone_lists):
        if wi > 0:
            out.append(word_delim)
        out.extend(phones)
    return out


# -----------------
# Audio loading
# -----------------
_resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}

def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)  # mono
    if sr != target_sr:
        if sr not in _resampler_cache:
            _resampler_cache[sr] = torchaudio.transforms.Resample(sr, target_sr)
        wav = _resampler_cache[sr](wav)
    x = wav.detach().cpu().numpy().astype(np.float32)
    return x


# -----------------
# Levenshtein on token lists
# -----------------
def edit_distance(a: List[str], b: List[str]) -> int:
    # DP with O(min(n,m)) memory
    if len(a) < len(b):
        a, b = b, a
    # now len(a) >= len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


# -----------------
# CTC greedy decode (keeps "|" as token if present)
# -----------------
def ctc_greedy_decode_tokens(logits: torch.Tensor, tokenizer, blank_id: int) -> List[List[str]]:
    """
    logits: (B, T, V)
    Returns: list of token lists, one per batch item
    """
    pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
    pred_ids = pred_ids.detach().cpu().numpy()

    out: List[List[str]] = []
    for seq in pred_ids:
        # collapse repeats
        collapsed = []
        prev = None
        for x in seq.tolist():
            if x == prev:
                continue
            collapsed.append(x)
            prev = x
        # remove blanks
        collapsed = [x for x in collapsed if x != blank_id]
        toks = tokenizer.convert_ids_to_tokens(collapsed)
        # remove special tokens if any slipped through
        toks = [t for t in toks if t not in ["<pad>", "<s>", "</s>"]]
        out.append(toks)
    return out


# -----------------
# Word-boundary metrics
# -----------------
def split_words(tokens: List[str], word_delim: str = WORD_DELIM) -> List[str]:
    """
    tokens: phoneme tokens possibly including word_delim
    returns list of "word tokens" where each word is a string of phonemes joined by space
    """
    words: List[List[str]] = [[]]
    for t in tokens:
        if t == word_delim:
            if words[-1]:  # only start new word if current non-empty
                words.append([])
            continue
        words[-1].append(t)
    # drop empties
    words = [w for w in words if w]
    return [" ".join(w) for w in words]


def boundary_positions(tokens: List[str], word_delim: str = WORD_DELIM) -> List[int]:
    """
    Represent each boundary by the cumulative #phones BEFORE it (excluding delimiters).
    """
    pos = []
    nphones = 0
    for t in tokens:
        if t == word_delim:
            pos.append(nphones)
        else:
            nphones += 1
    return pos


def boundary_prf(ref_tokens: List[str], hyp_tokens: List[str], word_delim: str = WORD_DELIM) -> Tuple[float, float, float]:
    ref = set(boundary_positions(ref_tokens, word_delim))
    hyp = set(boundary_positions(hyp_tokens, word_delim))
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    tp = len(ref.intersection(hyp))
    p = safe_div(tp, len(hyp))
    r = safe_div(tp, len(ref))
    f1 = safe_div(2*p*r, (p+r)) if (p+r) else 0.0
    return p, r, f1


# -----------------
# Evaluation
# -----------------
@dataclass
class UtteranceResult:
    audio_path: str
    grid_path: str
    ref_tokens: List[str]
    hyp_tokens: List[str]
    ref_len: int
    hyp_len: int
    edits: int


def evaluate_model(
    model_dir: Path,
    df: pd.DataFrame,
    ref_mode: str,
    batch_size: int,
    device: str,
    max_items: Optional[int] = None,
) -> Tuple[Dict[str, float], List[UtteranceResult]]:
    """
    ref_mode:
      - "flat": reference tokens from load_phones_flat
      - "words": reference tokens from load_phones_with_word_boundaries (with "|")
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id

    # Build references + audio paths list
    rows = df.to_dict("records")
    if max_items is not None:
        rows = rows[: max_items]

    refs: List[List[str]] = []
    audios: List[np.ndarray] = []
    audio_paths: List[str] = []
    grid_paths: List[str] = []

    for r in rows:
        ap = r["audio_path"]
        gp = r["grid_path"]
        audio_paths.append(ap)
        grid_paths.append(gp)
        audios.append(load_audio(ap, TARGET_SR))
        if ref_mode == "flat":
            refs.append(load_phones_flat(Path(gp), TIER_NAME))
        elif ref_mode == "words":
            refs.append(load_phones_with_word_boundaries(Path(gp), TIER_NAME, WORD_DELIM))
        else:
            raise ValueError(f"Unknown ref_mode: {ref_mode}")

    results: List[UtteranceResult] = []

    # Batched inference
    with torch.no_grad():
        for i in range(0, len(audios), batch_size):
            batch_audio = audios[i:i+batch_size]
            batch_refs = refs[i:i+batch_size]
            batch_audio_paths = audio_paths[i:i+batch_size]
            batch_grid_paths = grid_paths[i:i+batch_size]

            inputs = processor(batch_audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits  # (B,T,V)
            hyp_tok_lists = ctc_greedy_decode_tokens(logits, tokenizer, blank_id)

            for ap, gp, ref_toks, hyp_toks in zip(batch_audio_paths, batch_grid_paths, batch_refs, hyp_tok_lists):
                e = edit_distance(ref_toks, hyp_toks)
                results.append(
                    UtteranceResult(
                        audio_path=ap,
                        grid_path=gp,
                        ref_tokens=ref_toks,
                        hyp_tokens=hyp_toks,
                        ref_len=len(ref_toks),
                        hyp_len=len(hyp_toks),
                        edits=e,
                    )
                )

    # Aggregate core metrics
    total_edits = sum(r.edits for r in results)
    total_ref = sum(r.ref_len for r in results)
    per = safe_div(total_edits, total_ref)
    ser = safe_div(sum(1 for r in results if r.edits > 0), len(results))
    avg_ref_len = safe_div(total_ref, len(results))
    avg_hyp_len = safe_div(sum(r.hyp_len for r in results), len(results))

    metrics = dict(
        per=per,
        ser=ser,
        n_utterances=float(len(results)),
        avg_ref_len=avg_ref_len,
        avg_hyp_len=avg_hyp_len,
    )

    # If tokenizer contains WORD_DELIM, compute extra metrics from results
    if WORD_DELIM in getattr(tokenizer, "get_vocab", lambda: {})().keys() or WORD_DELIM in tokenizer.get_vocab():
        # PER excluding delimiters (if present in refs/hyps)
        per_no_delim_num = 0
        per_no_delim_den = 0

        # Word WER (split tokens on delimiter and compare word lists)
        wer_num = 0
        wer_den = 0

        # Boundary precision/recall/f1
        ps, rs, fs = [], [], []

        for r in results:
            ref_nd = [t for t in r.ref_tokens if t != WORD_DELIM]
            hyp_nd = [t for t in r.hyp_tokens if t != WORD_DELIM]
            per_no_delim_num += edit_distance(ref_nd, hyp_nd)
            per_no_delim_den += len(ref_nd)

            ref_words = split_words(r.ref_tokens, WORD_DELIM)
            hyp_words = split_words(r.hyp_tokens, WORD_DELIM)
            wer_num += edit_distance(ref_words, hyp_words)
            wer_den += len(ref_words)

            p, rr, f1 = boundary_prf(r.ref_tokens, r.hyp_tokens, WORD_DELIM)
            ps.append(p); rs.append(rr); fs.append(f1)

        metrics["per_no_delim"] = safe_div(per_no_delim_num, per_no_delim_den)
        metrics["wer_words"] = safe_div(wer_num, wer_den)
        metrics["boundary_p"] = float(np.mean(ps)) if ps else 0.0
        metrics["boundary_r"] = float(np.mean(rs)) if rs else 0.0
        metrics["boundary_f1"] = float(np.mean(fs)) if fs else 0.0

    return metrics, results


def write_per_utt_csv(out_csv: Path, results: List[UtteranceResult]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "audio_path", "grid_path",
            "ref_len", "hyp_len", "edits", "per_utt",
            "ref_tokens", "hyp_tokens",
        ])
        for r in results:
            per_utt = safe_div(r.edits, r.ref_len)
            w.writerow([
                r.audio_path, r.grid_path,
                r.ref_len, r.hyp_len, r.edits, per_utt,
                " ".join(r.ref_tokens), " ".join(r.hyp_tokens),
            ])


def print_top_errors(results: List[UtteranceResult], k: int = 10) -> None:
    scored = []
    for r in results:
        per_utt = safe_div(r.edits, r.ref_len)
        scored.append((per_utt, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"\nTop {min(k, len(scored))} worst utterances by PER:")
    for per_utt, r in scored[:k]:
        print("-" * 80)
        print(f"PER_utt={per_utt:.3f} edits={r.edits} ref_len={r.ref_len} hyp_len={r.hyp_len}")
        print(f"audio: {r.audio_path}")
        print(f"grid : {r.grid_path}")
        print("REF:", " ".join(r.ref_tokens[:200]))
        print("HYP:", " ".join(r.hyp_tokens[:200]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", type=str, default=str(DEFAULT_MODEL_A))
    ap.add_argument("--model_b", type=str, default=str(DEFAULT_MODEL_B))
    ap.add_argument("--test_csv", type=str, default=str(DEFAULT_TESTCSV))
    ap.add_argument("--audio_dir", type=str, default=str(DEFAULT_AUDIO_DIR))
    ap.add_argument("--grids_dir", type=str, default=str(DEFAULT_GRIDS_DIR))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto")
    ap.add_argument("--max_items", type=int, default=None, help="debug: evaluate only first N rows")
    args = ap.parse_args()

    model_a = Path(args.model_a)
    model_b = Path(args.model_b)
    test_csv = Path(args.test_csv)
    audio_dir = Path(args.audio_dir)
    grids_dir = Path(args.grids_dir)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Device:", device)
    print("Loading CSV:", test_csv)
    df = load_two_col_paths(test_csv, audio_dir, grids_dir)
    print("Rows:", len(df))

    # ---------
    # Model A: flat phones (no "|")
    # ---------
    print("\n=== Evaluating Model A (no word boundaries) ===")
    print("Model:", model_a)
    metrics_a, results_a = evaluate_model(
        model_dir=model_a,
        df=df,
        ref_mode="flat",
        batch_size=args.batch_size,
        device=device,
        max_items=args.max_items,
    )
    for k in ["per", "ser", "avg_ref_len", "avg_hyp_len", "n_utterances"]:
        print(f"{k:>12}: {metrics_a.get(k, 0):.6f}")
    out_a = model_a.parent / "eval_modelA_per_utt.csv"
    write_per_utt_csv(out_a, results_a)
    print("Per-utterance CSV written:", out_a)
    print_top_errors(results_a, k=8)

    # ---------
    # Model B: word boundaries (insert "|" in reference)
    # ---------
    print("\n=== Evaluating Model B (WITH word boundaries '|') ===")
    print("Model:", model_b)
    metrics_b, results_b = evaluate_model(
        model_dir=model_b,
        df=df,
        ref_mode="words",
        batch_size=args.batch_size,
        device=device,
        max_items=args.max_items,
    )
    # print key metrics
    keys_b = ["per", "per_no_delim", "wer_words", "boundary_p", "boundary_r", "boundary_f1", "ser", "avg_ref_len", "avg_hyp_len", "n_utterances"]
    for k in keys_b:
        if k in metrics_b:
            print(f"{k:>12}: {metrics_b.get(k, 0):.6f}")

    out_b = model_b.parent / "eval_modelB_per_utt.csv"
    write_per_utt_csv(out_b, results_b)
    print("Per-utterance CSV written:", out_b)
    print_top_errors(results_b, k=8)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
