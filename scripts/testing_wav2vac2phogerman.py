from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_DIR = Path(r"C:\PY\commonphone\CP\de\hf_wav2vec2_de_phonemes")
AUDIO_FILE = Path(r"C:\PY\commonphone\CP\de\wav\common_voice_de_17299216.wav")  # change me

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

target_sr = processor.feature_extractor.sampling_rate

wav, sr = torchaudio.load(str(AUDIO_FILE))
wav = wav.mean(dim=0)  # mono

if sr != target_sr:
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

inputs = processor(
    wav.numpy(),
    sampling_rate=target_sr,
    return_tensors="pt",
    padding=True,
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)

# decoded phoneme sequence (space-separated)
pred = processor.batch_decode(pred_ids)[0]
print("PRED:", pred)


