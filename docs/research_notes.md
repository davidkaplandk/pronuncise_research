### 1. Whisper

### 1.1 Architektur & Grundidee

- Modelltyp: **Encoder–Decoder-Transformer**.
- Pretraining: ~680k Stunden mehrsprachige, schwach annotierte Sprachdaten.
- Aufgaben: ASR, Speech Translation, Language ID.
- Implementation/Repo: (https://github.com/openai/whisper)

### 1.2 Funktionsweise

- Audio wird vom Encoder in latente Repräsentationen umgewandelt.
- Der Decoder generiert Token **autoregressiv** (Text oder andere Sequenzen).
- Im Standard-Setup ist das Ziel ein **orthografischer** Text (Subword-Tokenisierung).

### 1.3 Vor- und Nachteile für phonetische Anwendungen

**Vorteile**

- Sehr robuste, mehrsprachige Repräsentationen.
- Starker Decoder mit Sprachmodell-Eigenschaften:
  - Nutzt Kontext gut aus.
  - Gut bei „messy“ Realwelt-Audio.
- Kann theoretisch auf **beliebige Token** feinjustiert werden (z. B. IPA-Symbole statt Buchstaben).

**Nachteile**

- Stark **orthografie- und sprachmodell-basiert**:
  - “Normalisiert” oft hin zu Standardorthografie (Verzögerungslaute etc. werden in der Transkription oft ausgelassen)
  - Nicht ideal, wenn man „fehlerhafte“ oder stark dialektale Aussprache explizit sehen will.
- Kein intrinsisches Frame-Alignment (keine CTC-Zeitleisten wie bei Wav2Vec2).

### 1.4 Whisper für IPA / Phoneme

Ansatz (aus Papern und Diskussionen zusammengefasst):

1. Speech–Text-Daten sammeln.
2. Text mit einem **Phonemizer** (z. B. eSpeak, Phonemizer-Python) nach IPA konvertieren.
3. Neuen Tokenizer mit IPA-Vokabular bauen.
4. Embeddings/Output-Layer von Whisper an das neue Vokabular anpassen.
5. Whisper auf **Audio → IPA-Sequenzen** feinjustieren.

Relevante Ressourcen:

- Fine-Tuning mit IPA-Ziel (Beispielpaper): (https://arxiv.org/abs/2508.19270)  
- GitHub-Diskussion zur IPA-Ausgabe von Whisper: (https://github.com/openai/whisper/discussions/318)  
- Allgemeines Whisper-Fine-Tuning-Tutorial: (https://huggingface.co/blog/fine-tune-whisper)

---

## 2. Wav2Vec2

### 2.1 Architektur & Funktionsprinzip

- Modelltyp: **reiner Encoder** (Transformer) mit selbstüberwachtem Pretraining.
- Input: Roh-Audio (Waveform); Output: latente Sequenz von Feature-Vektoren.
- Typischer Downstream-Head: **CTC-Decoder**, der Token (z. B. Buchstaben oder Phoneme) für jedes Frame vorhersagt.

### 2.2 CTC-Decoder & Feature-Extraction

- **Feature-Extraction**:
  - Audio → normalisierte Frames → Encoder → versteckte Repräsentation.
- **CTC (Connectionist Temporal Classification)**:
  - Ermöglicht Sequenzvorhersage ohne exakten Frame-zu-Label-Align während des Trainings.
  - Dekodiert durch:
    - Best-Path-Decoding oder
    - Beam-Search mit Sprachmodell (optional).
- Wichtige Eigenschaft:
  - CTC-Output ist **frame-synchron** → geeignet für Alignment-Aufgaben.

### 2.3 Vor- und Nachteile für Phonetik

**Vorteile**

- Sehr gut geeignet für **phonem-genaue** Sequenzen mit Zeitinformation.
- CTC liefert eine natürliche Grundlage für **Forced Alignment** / Time-Stamps.
- Selbstüberwachtes Pretraining → gute Generalisierung, auch bei relativ wenig gelabelten Daten.

**Nachteile**

- Kein eingebauter Decoder mit LM wie bei Whisper; LM muss ggf. separat gebaut werden.
- Roh-CTC-Ausgabe ist noisy (Wiederholungen, „Buchstaben-Stottern“)

### 2.4 Wav2Vec2 für Phoneme / IPA

Ideen:

- Phonem-Modelle direkt von HuggingFace laden („wav2vec2 phoneme“ suchen).
- Eigenes Phonem-Vokabular definieren:
  - Label-Datei mit IPA-/Phonem-Symbolen.
  - CTC-Head neu initialisieren und auf (Audio, Phonem)-Daten trainieren.
- Beispiel-Tutorial: „Phoneme Recognition with Wav2Vec2“ (https://huggingface.co/blog/fine-tune-wav2vec2-english)  
  → zwar Englisch, aber methodisch relevant.

**Wav2Vec2-XLSR / XLS-R**

- Cross-lingual-Variante, auf vielen Sprachen vortrainiert.
- Gut geeignet für **multilinguale Phonemerkennung** (z. B. Allophant, Allosaurus).

---

## 3. Datensätze & Ressourcen (erste Notizen)

Potentiell relevante Datensätze:

- **Common Voice Deutsch**: (https://commonvoice.mozilla.org/)  
- **VoxPopuli** (mehrsprachig, EU-Parlamentsreden).
- **CSS10 German**: Einsprachiger TTS/Sprachdatensatz.


## 4. Kiel Corpus 

## Inhalt des Datensatzes

Je nach Teilkorpus umfasst das Kiel Corpus u. a.:

- **Gesprochene Sprache**  
  - Gelesene Sätze und Wörter  
  - Spontansprache (z. B. Dialoge, Erzählungen)  
- **Audio**  
  - Aufnahmen (WAV o. Ä.) mit definierten Aufnahmebedingungen  
- **Annotationen**  
  - Zeitlich ausgerichtete Label-Dateien (z. B. TextGrid, Tabellen/CSV)  
  - Orthografische, phonemische und prosodische Informationen

Nicht alle Teilkorpora haben exakt die gleichen Dateiformate oder Spalten, aber die Struktur folgt in der Regel dem gleichen Prinzip: **Signal + zeitlich ausgerichtete Annotationen**.

---

## Annotationsebenen

Typische Annotationsebenen im Kiel Corpus sind:

- **Segment-Ebene**  
  - Einzelne Laute/Phone mit Start- und Endzeit  
  - Symbolik meistens in einem maschinenlesbaren Phonem- oder Allophon-Inventar (nicht immer „saubere IPA“, sondern korpusinterne Notation)
- **Silben-Ebene**  
  - Silbengrenzen  
  - Betonungsinformationen (betont/unbetont)
- **Wort-Ebene**  
  - Orthografische Wortform  
  - Segmentierung in Wörter mit Zeitmarken  
- **Äußerungs-/Intonations-Ebene**  
  - Phrasengrenzen  
  - Intonationskonturen, Pausen, besondere prosodische Ereignisse (z. B. Akzente, Grenzton)

Welche Ebenen genau vorhanden sind, hängt vom jeweiligen Unterkorpus ab.

---

## Typische Spalten in den Annotationstabellen

In tabellarischen Annotationen (z. B. TXT/CSV) kommen u. a. folgende Spalten vor (konkrete Namen können variieren):

- `file` / `utt_id` – ID der Äußerung/Datei  
- `channel` – Kanalnummer (falls Stereo/Dialog)  
- `start_time` – Segment-/Wortbeginn in Sekunden  
- `end_time` – Segment-/Wortende in Sekunden  
- `label` – Segment- oder Wortlabel  
  - bei Segmenten: Phonem/Allophon  
  - bei Wörtern: orthografische Wortform  
- `tier` / `level` – Ebene der Annotation (Segment, Wort, Silbe, Phrase …)  
- `speaker` – Sprecherkennzeichnung  
- optional prosodische Spalten, z. B.:  
  - `stress` – Betonungsgrad  
  - `break_index` – Phrase Break / Grenzstärke  
  - `accent` – Art des Akzents  
  - weitere korpusinterne Kategorien (Tonhöhenmuster etc.)

Für die eigene Auswertung wählt man meist:

1. **Relevante Ebene** (z. B. Segment-Ebene für Phone-Statistiken, Wort-Ebene für Dauer & Wortlisten)  
2. **Zeitsäulen** (`start_time`, `end_time`) zur Alignierung mit dem Audio  
3. **Labelspalte** (`label` o. ä.) als Zielgröße (Phone/Wort/Prosodie)

---

## Entstehung und Design

Das Kiel Corpus wurde in mehreren Projekten über Jahre hinweg aufgebaut. Typische Schritte bei der Erstellung:

1. **Aufnahme**  
   - Rekrutierung von Muttersprachler:innen des Deutschen  
   - Aufnahme unter kontrollierten Bedingungen (z. B. ruhige Studio-Umgebung)

2. **Grundtranskription**  
   - Orthografische Transkription der Aufnahmen  
   - Segmentierung in Äußerungen und Sätze

3. **Phonetische/phonologische Annotation**  
   - Manuelle oder halbautomatische Segmentierung in Phone/Silben  
   - Verwendung eines festgelegten Lautinventars (teilweise korpusspezifische Symbole)

4. **Prosodische Annotation**  
   - Markierung von Betonung, Akzenten, Phrasengrenzen und Pausen  
   - Nutzung definierter Kategorien und Richtlinien

Damit ist das Korpus **zeitlich fein ausgerichtet** und eignet sich gut für Studien zu Lautdauer, Koartikulation, Intonation, Rhythmus usw.

---

## Nutzen für Pronuncise

- **ASR/Phone-Modelle**  
  - Nutzung der Segmentlabels als Goldstandard für Phone-Erkennung  
  - Training oder Evaluation von Modellen 


---

## TL;DR

- Das **Kiel Corpus** ist ein deutschsprachiges, phonetisch und prosodisch annotiertes Sprachkorpus mit Audio + zeitlich ausgerichteten Labels.  
- Es bietet Annotationen auf **Segment-, Silben-, Wort- und Prosodie-Ebene**, mit Spalten wie `start_time`, `end_time`, `label`, `speaker` usw.  
- Die Lautnotation ist **korpusintern** (nicht immer pures IPA), deshalb unbedingt die Beschreibung der Symbole lesen.  
- Ideal für Phone-/ASR-Evaluation, prosodische Analysen und phonetische Detailstudien.
