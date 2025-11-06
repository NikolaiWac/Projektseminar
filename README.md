# Projektseminar – Einfaches neuronales Netz (Java)

Projektstruktur:
- ActFuntions.java
  - Enthält einfache ActFuntions: Identität (0), Step (1, Schwelle aktuell 0), Sigmoid (2, Default).
  - funktionSelect(x, aktFkt) wählt anhand einer Ganzzahl die Funktion.
- Neuron.java
  - Felder: aktFkt (Aktivierungs-ID), weights (Gewichte).
  - outputFkt(input): gewichtete Summe (Bias derzeit 0), danach Aktivierung. Achtung: asignrandomWeights() setzt bei jedem Aufruf zufällige Gewichte → nicht deterministisch, kein Training implementiert.
- Schicht.java
  - Hält eine Liste von Neuronen. schichtSum(input) summiert die Ausgaben aller Neuronen der Schicht.
- Netz.java
  - Baut das Netz aus mehreren Schichten (Konstruktor mit Neuronenanzahl je Schicht).
  - vorwaerts(eingabe): propagiert durch die Schichten, wobei pro Schicht die Summenbildung erfolgt und als Einzeleingabe weitergereicht wird.
  - main: einfacher Konsolen‑Dialog zur Erstellung eines Netzes und Berechnung eines Outputs.
  - Hilfssetter: setNeuronFkt(layer, pos, fkt), setNeuronWeights(layer, pos, inputPos, weight).

Schnellstart:
- In IntelliJ IDEA: Projekt öffnen und Netz.main ausführen.
- Über Konsole (ohne Pakete):
  - javac -d out src\*.java
  - java -cp out Netz

Hinweise/Limitierungen:
- Kein Lernen/Backpropagation; Gewichte werden aktuell zufällig gesetzt, daher ändert sich der Output bei jedem Lauf.
- Bias kann in mittlerweile benutzt werden, Step‑Schwelle dynamisch setzbar aber Standard auf 0.