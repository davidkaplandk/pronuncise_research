# Woche 1 – Summary

## 1. Setup & Infrastruktur

- Python-Umgebung eingerichtet, zentrale Pakete installiert:
  - `torch`, `torchaudio`, `transformers`, 
- Projektstruktur (geplant/teilweise umgesetzt):
  - `docs/`, `scripts/`, `test_audio/`, `output/`.
- Common Voice German samples installiert

---

## 2. Grundlagen: Phonetische Transkription & IPA

- Unterschieden zwischen:
  - **phonetisch** (fein, allophon) vs. **phonemisch** (kontrastive Einheiten).
- Erste IPA-Skizze für Deutsch (Konsonanten, Vokale).
- Motivation: Orthografische ASR glättet Aussprache; für Aussprachebewertung und Forschung werden explizite Phoneme/IPA benötigt.

---

## 3. Erste Tests

- `test_whisper.py`:
  - Whisper auf deutsche Test-Audios angewendet.
  - Eindruck: starke Normalisierung auf Standardorthografie, feine Ausspracheunterschiede kaum sichtbar.

- `test_wav2vec2.py`:
  - Deutsches Wav2Vec2-CTC-Modell geladen, Features und Buchstaben-CTC-Output inspiziert.
  - Beobachtung: typische CTC-Wiederholungen, nach Collapse sinnvolle Sequenzen; für Phoneme später Vokabularwechsel nötig.

---

## 4. Fazit & Ausblick

- Beide Modelle prinzipiell geeignet:
  - Whisper → sequenzielle IPA-Ausgabe über Fine-Tuning.
  - Wav2Vec2 → zeitaufgelöste Phoneme via CTC.
- Nächste Schritte (Woche 2):
  - Datensätze systematisch erfassen (`docs/datasets.md`).
  - MFA installieren und an kleinen Beispielen testen.
  - Erste phonetische Modelle (Allosaurus / Wav2Vec2-Phonem) ausprobieren und Outputs sammeln.
  - PER-Implementierung für erste Evaluation vorbereiten.

**TL;DR:** Setup steht, erste Audios und Basistests mit Whisper & Wav2Vec2 sind gemacht. Whisper wirkt stark normalisierend, Wav2Vec2 ist vielversprechend für Phoneme + Zeitstempel. Woche 2 fokussiert auf Datensätze, Forced Alignment und erste phonetische Modelle.
