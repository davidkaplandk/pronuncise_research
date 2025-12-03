# Wochenbericht – Woche 2

## Überblick

In Woche 2 habe ich mich vor allem mit der **Evaluation von Modellen** und der **Suche nach geeigneten Korpora** beschäftigt. Ein Schwerpunkt lag auf der technischen Einrichtung und dem Testen verschiedener Pipelines (ASR/Phonemisierung/Alignment).

---

## Modell-Evaluation

### Wav2Vec-Modelle

- Ich habe versucht, **Wav2Vec-basierte Modelle** für die phonemische bzw. phonologische Auswertung zu verwenden.
- Dabei ist ein **Bug in meiner Wav2Vec-Pipeline** aufgetreten, den ich bisher noch nicht beheben konnte.
  - In Folge dessen ist die geplante systematische Evaluation (z. B. Berechnung von Phone Error Rate) aktuell noch **nicht zuverlässig** möglich.
  - Die bisherigen Ergebnisse aus dieser Pipeline sind daher **nicht vertrauenswürdig** und können im Moment **nicht als valide Evaluation** verwendet werden.

### Verschiedene Modelle & inkompatible Phone-Sets

- Zusätzlich wollte ich **verschiedene Modelle** (z. B. Allosaurus, Neuralang, eSpeak/phonemizer, Wav2Vec-Varianten) **systematisch miteinander vergleichen**.
- In der Praxis hat sich aber gezeigt, dass eine saubere Evaluation aktuell **nahezu unmöglich** ist, weil:
  - jedes Modell ein **anderes IPA-/Phone-Inventar** verwendet (unterschiedliche Symbole für dieselben Laute),
  - sich die **Segmentierung** unterscheidet (z. B. Diphthonge vs. zwei Vokale, Affrikaten vs. zwei Segmente),
  - teilweise **Stressmarkierungen, Längenzeichen oder Sonderzeichen** inkonsistent gesetzt werden.
- Dadurch sind direkte Kennzahlen wie **Phone Error Rate** zwischen den Systemen **nicht sinnvoll interpretierbar**, solange kein gemeinsames, kanonisches Phone-Set und kein robustes Mapping definiert sind.

### Montreal Forced Aligner

- Ich habe den **Montreal Forced Aligner (MFA)** eingerichtet und erfolgreich laufen lassen.
- Ziel ist es, damit **zeitlich ausgerichtete Phone-/Wort-Alignments** zu erhalten, die sich später für:
  - Referenz-Alignments (Goldstandard) und
  - Qualitätssicherung der automatischen Transkriptionen
  nutzen lassen.

---

## Korpora / Datensätze

### Kiel Corpus

- Ich habe den **Kiel Corpus** gefunden und mir angeschaut.
- Der Korpus ist für **phonetische und prosodische Analysen** interessant (deutsches Material, fein annotiert).
- Perspektivisch könnte er:
  - als **Referenzkorpus** für phonemische Auswertungen,
  - zur **Validierung von Alignments** und
  - für methodische Experimente (z. B. Mapping von IPA/Phone-Sets)
  genutzt werden.

### FOLK Corpus

- Zusätzlich habe ich den **FOLK Corpus** (Forschungs- und Lehrkorpus Gesprochenes Deutsch) identifiziert.
- Nach aktuellem Stand scheint der FOLK-Corpus jedoch **nur für Forschungszwecke** nutzbar zu sein, teilweise mit **restriktiveren Lizenzbedingungen**.
- Für mein Projekt bedeutet das:
  - Nutzung ist ggf. nur im **engen wissenschaftlichen Rahmen** möglich,
  - als **Trainings- oder frei verwendbarer Evaluationskorpus** ist er wahrscheinlich weniger geeignet.

---

