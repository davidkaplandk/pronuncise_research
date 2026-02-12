import json
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from datasets import Dataset
import evaluate

from praatio import textgrid as praatio_textgrid
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from transformers.models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
import inspect
from transformers import Trainer

# -----------------
# Paths
# -----------------
BASE = Path(r"C:\PY\commonphone\CP\de")
TRAIN_CSV = BASE / "train_wav.csv"
TEST_CSV  = BASE / "test_wav.csv"

AUDIO_DIR = BASE / "wav"
GRIDS_DIR = BASE / "grids"

OUT_DIR   = BASE / "hf_wav2vec2_de_phonemes"
TARGET_SR = 16000




# -----------------
# CSV loading (STRICT: use only audio_file + grid_file)
# -----------------
def load_two_col_paths(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize headers: strip spaces, remove BOM, spaces->underscore, lower
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "audio_file" not in df.columns:
        raise ValueError(f"'audio_file' not found. Columns are: {list(df.columns)}")
    if "grid_file" not in df.columns:
        raise ValueError(f"'grid_file' not found. Columns are: {list(df.columns)}")

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

    df["audio_path"] = df["audio_file"].apply(lambda x: to_abs(x, AUDIO_DIR))
    df["grid_path"]  = df["grid_file"].apply(lambda x: to_abs(x, GRIDS_DIR))
    return df


# -----------------
# TextGrid -> phoneme tokens
# -----------------
def load_phones_from_textgrid(tg_path: Path, tier_name: str = "KAN-MAU"):
    tg = praatio_textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

    if tier_name not in tg.tierNames:
        raise ValueError(f"{tg_path}: missing tier '{tier_name}'. Available: {tg.tierNames}")

    tier = tg.getTier(tier_name)

    phones = []
    for (_, _, label) in tier.entries:
        lab = (label or "").strip()
        if lab == "":          # silence in your TextGrids
            continue

        # KAN-MAU stores phones as "ɡ eː ɡ ə n ..."
        tokens = lab.split()
        phones.extend(tokens)

    if not phones:
        raise ValueError(f"{tg_path}: extracted 0 phones from {tier_name}")

    return phones


# -----------------
# Load CSVs -> datasets
# -----------------
train_df = load_two_col_paths(TRAIN_CSV)
test_df  = load_two_col_paths(TEST_CSV)

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)



def add_phonemes(batch):
    batch["phonemes"] = load_phones_from_textgrid(Path(batch["grid_path"]), tier_name="KAN-MAU")
    return batch


train_ds = train_ds.map(add_phonemes)
test_ds  = test_ds.map(add_phonemes)

print("Example phones:", train_ds[0]["phonemes"][:30])

# -----------------
# Vocab + processor
# -----------------
all_phones = set()
for phones in train_ds["phonemes"]:
    all_phones.update(phones)

special = ["<pad>", "<unk>", "<s>", "</s>"]
vocab = {tok: i for i, tok in enumerate(special)}
for p in sorted(all_phones):
    if p not in vocab:
        vocab[p] = len(vocab)

OUT_DIR.mkdir(parents=True, exist_ok=True)
vocab_path = OUT_DIR / "vocab.json"
with vocab_path.open("w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

tokenizer = Wav2Vec2PhonemeCTCTokenizer(
    str(vocab_path),
    do_phonemize=False,
    phone_delimiter_token=" ",
    pad_token="<pad>",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
)

### unknown signs 
unk_id = vocab["<unk>"]
n_unk = sum(1 for p in all_phones if tokenizer.convert_tokens_to_ids(p) == unk_id)
print("Unknown phones in vocab mapping:", n_unk)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=TARGET_SR,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# -----------------
# Audio loading + prepare batches
# -----------------
resampler_cache = {}

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)  # mono
    if sr != TARGET_SR:
        if sr not in resampler_cache:
            resampler_cache[sr] = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = resampler_cache[sr](wav)
    return wav.numpy()


def prepare_batch(batch):
    speech = load_audio(batch["audio_path"])
    batch["input_values"] = processor(speech, sampling_rate=TARGET_SR).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # ✅ No tokenizer(...) call -> avoids return_offsets_mapping issue
    batch["labels"] = tokenizer.convert_tokens_to_ids(batch["phonemes"])
    return batch




train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names)
test_ds  = test_ds.map(prepare_batch, remove_columns=test_ds.column_names)


class DataCollatorCTC:
    def __init__(self, processor, tokenizer, padding=True):
        self.processor = processor
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTC(processor, tokenizer)


# -----------------
# Metric
# -----------------
wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    return {"per": wer.compute(predictions=pred_str, references=label_str)}


# -----------------
# Model + training
# -----------------
pretrained = "facebook/wav2vec2-large-xlsr-53"

model = Wav2Vec2ForCTC.from_pretrained(
    pretrained,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
)
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()  # Reduces memory usage on weak GPU

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    group_by_length=True,
    length_column_name="input_length",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,     # also lower eval batch
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    prediction_loss_only=True,        # ✅ key fix
    num_train_epochs=10,
    fp16=True,
    save_steps=500,
    eval_steps=2000,
    logging_steps=100,
    logging_first_step=True,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    report_to="none",
)


trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,  # disabled because prediction_loss_only=True
)


sig = inspect.signature(Trainer.__init__).parameters
if "processing_class" in sig:
    trainer_kwargs["processing_class"] = processor.feature_extractor
elif "tokenizer" in sig:
    trainer_kwargs["tokenizer"] = processor.feature_extractor  # older Trainer
# else: nothing to pass (see fallback below)

trainer = Trainer(**trainer_kwargs)

print(">>> starting training")
trainer.train()
print(">>> training finished")

processor.save_pretrained(str(OUT_DIR))
trainer.save_model(str(OUT_DIR))
print("Saved to:", OUT_DIR)
