# Kurzreport: Evaluation zweier Wav2Vec2-CTC Phonemmodelle (Deutsch, CommonPhone)

## Setup und Daten
Ziel war der Vergleich zweier akustischer CTC-Modelle zur **Phonemtranskription** auf Deutsch. Beide Modelle basieren auf *Wav2Vec2-Large-XLSR-53* und wurden auf CommonPhone-Daten trainiert. Als Evaluationsgrundlage wurden **1525** Audiodateien aus `test_wav.csv` verwendet, der Datensatz stammt von **CommonPhone** ([Zenodo: 5846137](https://zenodo.org/records/5846137)). Die Referenzlabels stammen aus den zugehörigen **Praat TextGrids** (Tier `KAN-MAU`) und enthalten phonemische Token.

**Wichtig:** `test_wav.csv` wurde **nicht für Gradienten-Updates** genutzt (also kein “Training” im engeren Sinn), aber **während des Trainings als Eval-Set** verwendet (Evaluation in Steps). Streng genommen ist es daher eher ein **Validierungsset** als ein vollkommen “unberührtes” Testset.

## Modelle
- **Modell A (Baseline):** Phonem-CTC ohne explizite Wortgrenzen. Ausgabe ist eine Folge von Phonemtokens (durch Leerzeichen getrennt), aber ohne “echte” Worttrennung.
- **Modell B (mit Wortgrenzen):** Gleiches Grundmodell, aber mit zusätzlichem Wortgrenzen-Token `|` in den Labels. Beim Decoding wird `|` als Worttrenner interpretiert (entspricht “Spaces zwischen Wörtern”).

## Metriken
Für beide Modelle wurde **greedy CTC decoding** verwendet (argmax → collapse repeats → remove blank). Als Hauptmetrik dient die **Phoneme Error Rate (PER)**, definiert als Levenshtein-Distanz auf Tokenebene geteilt durch die Anzahl Referenztoken. Zusätzlich wird **SER** (Sequence Error Rate) berichtet: Anteil der Utterances mit mindestens einem Tokenfehler.

### Ergebnisse (aus der Evaluation)
**Modell A**
- PER = **0.0566**
- SER = **0.7331**
- Ø Referenzlänge ≈ **47.8** Tokens

**Modell B (mit `|`)**
- PER (inkl. `|`) = **0.0540**
- **PER_no_delim** (ohne `|`, fairer Vergleich zu Modell A) = **0.0554**
- SER = **0.7626**
- Ø Referenzlänge ≈ **55.7** Tokens (höher wegen zusätzlicher `|`-Tokens)
- **WER_words** (Wortfehler auf Basis von `|`-Segmentierung) = **0.1944**
- **Boundary F1** (Qualität der Wortgrenzenpositionen) = **0.7892**
  - Precision ≈ 0.7889
  - Recall ≈ 0.7920

## Interpretation
1. **Phonemgenauigkeit (fairer Vergleich):**  
   Der relevante Vergleich ist **Modell A PER** vs. **Modell B PER_no_delim**, da Modell B zusätzliche Wortgrenzentokens enthält. Hier zeigt Modell B eine **leichte Verbesserung**:  
   `0.0566 → 0.0554` (≈ 2% relative Verbesserung).  
   Fazit: Wortgrenzen “kosten” die Phonemerkennung nicht, sondern sind praktisch kompatibel mit ähnlicher oder minimal besserer phonemischer Performance.

2. **Warum ist SER bei Modell B höher?**  
   SER zählt eine Utterance bereits als “falsch”, sobald **irgendein** Token nicht stimmt. Modell B hat **mehr Tokens** (wegen `|`) und damit mehr Möglichkeiten für kleine Fehler. Deshalb ist SER zwischen A und B **nicht direkt vergleichbar** und sollte nicht überbewertet werden.

3. **Qualität der Worttrennung:**  
   Modell B liefert zusätzlich eine brauchbare Wortsegmentierung: **Boundary F1 ≈ 0.79** deutet darauf hin, dass Wortgrenzen in der Mehrzahl korrekt gesetzt werden. Gleichzeitig zeigt **WER_words ≈ 0.19**, dass ungefähr jede fünfte Wort-Einheit (gemessen auf Wortebene) eine Edit-Operation benötigt. Für viele Downstream-Anwendungen (z. B. Vorsegmentierung, Alignment-Pipelines oder grobe Wortgrenzen) ist das bereits nützlich; für perfekte Wortgrenzen wäre noch Optimierung nötig (z. B. bessere Worttier-Labels, stärkeres Sprachmodell oder Fine-Tuning auf saubere Wortgrenzen).

## Fehleranalyse (qualitativ)
Die “Worst Cases” zeigen typische Muster: **Insertionen/Substitutionen** von Phonemen sowie teilweise Verwechslungen in kurzen oder schwierigen Segmenten. Auffällig ist, dass dieselben Audiodateien in beiden Modellen zu den schlechtesten Beispielen gehören, was auf **schwierige Audioqualität, starke Akzente oder Alignment-Probleme im TextGrid** hindeuten kann.

## Fazit
Beide Modelle erreichen eine niedrige PER (~5–6%). Modell B bietet zusätzlich Wortgrenzen mit guter Boundary-Qualität (F1 ~0.79) und **ohne nennenswerten Verlust** bei der phonemischen Genauigkeit (PER_no_delim sogar minimal besser). Für Anwendungen, die **Wort-segmentierte Phonemsequenzen** benötigen, ist Modell B klar vorzuziehen; für reine Phonemerkennung ohne Wortstruktur genügt Modell A.
