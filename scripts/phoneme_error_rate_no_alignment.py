import os
import pandas as pd
import numpy as np

# ======== PATHS (RELATIVE TO THIS SCRIPT) ========

script_dir = os.path.dirname(os.path.abspath(__file__))  # .../pronuncise_research/scripts
project_root = os.path.dirname(script_dir)               # .../pronuncise_research

test_sets_dir = os.path.join(project_root, "test_sets")
data_dir      = os.path.join(project_root, "data")

# --- auto-detect golden CSV in test_sets ---
csv_candidates = [
    f for f in os.listdir(test_sets_dir)
    if f.lower().endswith(".csv")
]

if not csv_candidates:
    raise FileNotFoundError(f"No CSV files found in {test_sets_dir}")

golden_name = None
for f in csv_candidates:
    if "golden" in f.lower():
        golden_name = f
        break

if golden_name is None:
    golden_name = csv_candidates[0]  # fallback to first CSV

GOLDEN_PATH = os.path.join(test_sets_dir, golden_name)
ALLO_PATH   = os.path.join(data_dir, "whisper_phonemes3.csv")

print("Using GOLDEN_PATH:", GOLDEN_PATH, "exists:", os.path.exists(GOLDEN_PATH))
print("Using ALLO_PATH:  ", ALLO_PATH,  "exists:", os.path.exists(ALLO_PATH))

# Adjust these to your real column names if needed
GOLD_ID_COL    = "id"
GOLD_IPA_COL   = "ipa"             # ideal IPA transcription in golden file
ALLO_ID_COL    = "id"
ALLO_TRANS_COL = "phonemes"   # allosaurus transcription column


# ======== HELPERS ========

def normalize_phone_string(s: str) -> str:
    s = str(s)
    s = s.replace("|", " ")
    s = " ".join(s.split())
    return s.strip()


def string_to_phone_list(s: str):
    s = normalize_phone_string(s)
    if not s:
        return []
    return s.split(" ")


def edit_distance(ref, hyp) -> int:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]


def jaro_similarity(a, b):
    a = list(a)
    b = list(b)
    len1, len2 = len(a), len(b)

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    a_matches = [False] * len1
    b_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    transpositions /= 2

    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a, b, p=0.1, max_prefix=4):
    j = jaro_similarity(a, b)

    prefix_len = 0
    for x, y in zip(a, b):
        if x == y:
            prefix_len += 1
            if prefix_len == max_prefix:
                break
        else:
            break

    return j + prefix_len * p * (1.0 - j)


# ======== LOAD DATA ========

gold = pd.read_csv(GOLDEN_PATH)
allo = pd.read_csv(ALLO_PATH)

merged = gold.merge(
    allo[[ALLO_ID_COL, ALLO_TRANS_COL]],
    left_on=GOLD_ID_COL,
    right_on=ALLO_ID_COL,
    how="inner",
    suffixes=("_gold", "_allo")
)

if merged.empty:
    raise ValueError("No overlapping IDs between golden and allosaurus CSVs!")

print(f"Found {len(merged)} matching utterances to evaluate.")

# ======== COMPUTE METRICS ========

per_utterance = []
total_errors = 0
total_phones = 0
jw_sum = 0.0
j_sum = 0.0
num_utts = 0

for _, row in merged.iterrows():
    utt_id = row[GOLD_ID_COL]

    ref_str = row[GOLD_IPA_COL]
    hyp_str = row[ALLO_TRANS_COL]

    ref_phones = string_to_phone_list(ref_str)
    hyp_phones = string_to_phone_list(hyp_str)

    if len(ref_phones) == 0:
        print(f"[WARN] Skipping ID {utt_id}: empty reference phones.")
        continue

    errors = edit_distance(ref_phones, hyp_phones)
    per = errors / len(ref_phones)
    norm_denom = max(len(ref_phones), len(hyp_phones)) or 1
    lev_norm_max = errors / norm_denom

    j = jaro_similarity(ref_phones, hyp_phones)
    jw = jaro_winkler(ref_phones, hyp_phones)

    total_errors += errors
    total_phones += len(ref_phones)
    jw_sum += jw
    j_sum += j
    num_utts += 1

    per_utterance.append({
        "id": utt_id,
        "ref_len": len(ref_phones),
        "hyp_len": len(hyp_phones),
        "errors_lev": errors,
        "PER_ref": per,
        "LEV_norm_max": lev_norm_max,
        "jaro": j,
        "jaro_winkler": jw,
        "ref_phones": " ".join(ref_phones),
        "hyp_phones": " ".join(hyp_phones),
    })

overall_per = total_errors / total_phones if total_phones > 0 else np.nan
mean_jaro = j_sum / num_utts if num_utts > 0 else np.nan
mean_jw   = jw_sum / num_utts if num_utts > 0 else np.nan

print("\n===== PHONE ERROR RATE (PER) & OTHER METRICS =====")
print(f"Utterances evaluated:     {num_utts}")
print(f"Total phones (reference): {total_phones}")
print(f"Total errors (S+I+D):     {total_errors}")
print(f"Overall PER (errors/ref): {overall_per:.4f} ({overall_per*100:.2f}%)")
print(f"Mean Jaro similarity:     {mean_jaro:.4f}")
print(f"Mean Jaro–Winkler:        {mean_jw:.4f}")

# ======== SAVE EVAL CSVs IN eval/ UNDER PROJECT ROOT ========

eval_dir = os.path.join(project_root, "eval")
os.makedirs(eval_dir, exist_ok=True)

details_out = os.path.join(eval_dir, "phone_error_details.csv")
summary_out = os.path.join(eval_dir, "phone_eval_summary.csv")

pd.DataFrame(per_utterance).to_csv(details_out, index=False, encoding="utf-8")

summary_df = pd.DataFrame([{
    "utterances": num_utts,
    "total_phones_ref": total_phones,
    "total_errors_lev": total_errors,
    "overall_PER_ref": overall_per,
    "mean_jaro": mean_jaro,
    "mean_jaro_winkler": mean_jw,
}])
summary_df.to_csv(summary_out, index=False, encoding="utf-8")

print(f"\nPer-utterance metrics saved to: {details_out}")
print(f"Summary metrics saved to:      {summary_out}")
