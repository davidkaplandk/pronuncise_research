# Pronuncise Research

Dieses Repository sammelt erste Experimente zur **phonetischen / phonemischen Spracherkennung** mit modernen Speech-Modellen (v. a. Whisper und Wav2Vec2).  
Ziel ist es, gesprochene Sprache nicht nur in Orthografie, sondern in **IPA-Phonemen** darzustellen und verschiedene Ansätze zu vergleichen.

## Ziele

- Arbeitsumgebung für Speech-Experimente aufsetzen
- Test-Audio („Golden Test Set“) definieren
- Erste Baselines mit:
  - Whisper / WhisperX
  - Wav2Vec2 (Text- und Phonem-Varianten)
  - ggf. Allosaurus / Forced Alignment (MFA)
- Später: Evaluationsmetriken (v. a. Phone Error Rate, PER)

## Voraussetzungen

- Python ≥ 3.10


Beispiel (Windows / PowerShell):

```powershell
cd C:\PY
git clone https://github.com/davidkaplandk/pronuncise_research.git
cd pronuncise_research

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt   # falls vorhanden
