mfatowav2vec2# Wortweise Ausrichtung mit MFA und phonemischem ASR

Dieses Skript implementiert eine End-to-End-Pipeline, um **deutsche Spontansprache** auf Wortebene zeitlich auszurichten und dabei **drei verschiedene Repräsentationen** zu erzeugen:

1. **Kanonische IPA-Transkription pro Wort** (aus der Orthographie via `phonemizer` + eSpeak-NG)  
2. **Telefonfolge aus dem Montreal Forced Aligner (MFA)** (Wörter und Phones aus dem TextGrid)  
3. **Telefonfolge aus einem phonemischen ASR-Modell** (`facebook/wav2vec2-lv-60-espeak-cv-ft`) für jeden Wort-Audioclip

Am Ende entsteht eine CSV-Datei mit **einer Zeile pro Wort**, die alle Informationen bündelt.

---

## 1. Pipeline-Überblick

Das Skript folgt grob diesen Schritten:

1. **(Optional) Alignment mit MFA**
   - Ruft `mfa align` auf einem Korpus auf (`mfa_corpus_german`).
   - Erwartet:
     - WAV-Dateien: `<utt_id>.wav`
     - Transkriptionen: `<utt_id>.lab` oder `<utt_id>.txt`
   - Verwendet ein deutsches MFA-Lexikon (`german_mfa.dict`) und ein deutsches Akustikmodell (`german_mfa.zip`).
   - Schreibt pro Äußerung ein TextGrid nach `mfa_output`.

2. **Einlesen der MFA-TextGrids**
   - Öffnet jedes `.TextGrid` aus `mfa_output`.
   - Extrahiert:
     - ein **Word-Tier** (`words` oder `word`),  
     - ein **Phone-Tier** (`phones` oder `phone`).
   - Speichert für jedes Intervall:
     - Startzeit `minTime`
     - Endzeit `maxTime`
     - Label (Wort oder Phone)

3. **Laden der Orthographie**
   - Liest die Transkription pro Äußerung aus `<utt_id>.lab` oder `<utt_id>.txt` in `mfa_corpus_german`.
   - Tokenisiert die Transkription sehr einfach per `text.split()` in eine Wortliste.

4. **Kanonische IPA-Transkription (Text → IPA)**
   - Verwendet `phonemizer` mit Backend `espeak` und Sprachcode `de`.
   - Für jedes Wort wird eine **kanonische IPA-Folge** erzeugt.
   - Die IPA-Phones werden mit Leerzeichen getrennt.

5. **Zuordnung MFA-Phones zu jedem Wort**
   - Für jedes Wortintervall (Start `w_start`, Ende `w_end`) im TextGrid werden alle Phone-Intervalle gesucht, deren Zeiten innerhalb dieses Wortfensters liegen.
   - Die zugehörigen Phone-Labels werden eingesammelt und als `mfa_phones` gespeichert (Leerzeichen-getrennte Folge).

6. **Audiobasiertes phonemisches ASR pro Wortsegment**
   - Lädt die WAV-Datei (mono, resampled auf 16 kHz) mit `torchaudio`.
   - Schneidet für jedes Wort das zugehörige Audiostück aus:
     - `start_sample = round(w_start * sr)`
     - `end_sample = round(w_end * sr)`
   - Übergibt dieses Segment an das phonemische Wav2Vec2-Modell
     `facebook/wav2vec2-lv-60-espeak-cv-ft`.
   - Dekodiert die Modell-Ausgabe zu einer Phonemsequenz (`asr_phones`).

7. **Export nach CSV**
   - Für jedes Wort wird eine Zeile in `data/aligned_words_ipa.csv` geschrieben.
   - Sie enthält:
     - Metadaten zur Äußerung und zum Wort
     - Zeitinformationen
     - Kanonische IPA-Transkription (`ipa_canonical`)
     - Phone-Sequenz aus MFA (`mfa_phones`)
     - Phone-Sequenz aus ASR-Modell (`asr_phones`)

---

