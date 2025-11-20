import csv
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll") #ad your path to libespeak-ng.dll !!!

# Input and output paths
in_csv = r"paths\to\your\transcriptions.csv"    # change for wanted file    
out_csv = r"paths\to\your\transcriptions_with_ipa.csv"  #change for wanted file 

rows = []
texts = []

# --- 1) Read existing CSV ---
with open(in_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    if "text" not in fieldnames:
        raise ValueError("Expected a 'text' column in the CSV.")

    for row in reader:
        rows.append(row)
        texts.append(row["text"])

print(f"Loaded {len(rows)} rows from {in_csv}")

# --- 2) Phonemize all texts in one go (much faster) ---
# German: language='de'
# Separator: phones separated by space, words by '|'
separator = Separator(phone=' ', word='|')

ipas = phonemize(
    texts,
    language="de",      
    backend="espeak",
    separator=separator,
    strip=True,
    preserve_punctuation=False,
    with_stress=True,     
    njobs=1            
)

# --- 3) Attach IPA to rows ---
for row, ipa in zip(rows, ipas):
    row["ipa"] = ipa

# --- 4) Write new CSV with extra 'ipa' column ---
out_fieldnames = fieldnames + ["ipa"]

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote IPA-augmented CSV to {out_csv}")
