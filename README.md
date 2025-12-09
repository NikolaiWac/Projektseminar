# Projektseminar – Neuronales Netz zur Ziffernerkennung (Java)

Dieses Projekt implementiert ein vollständig funktionsfähiges neuronales Netzwerk in Java ohne die Verwendung externer ML-Bibliotheken. Der aktuelle Fokus liegt auf der Erkennung handgeschriebener Ziffern mithilfe des MNIST-Datensatzes.

Das Netzwerk unterstützt mittlerweile Backpropagation, das Laden externer Daten (MNIST) und die Verarbeitung eigener PNG-Bilder.

## **Features**
- Multilayer Perceptron (MLP): Flexibler Aufbau von Schichten (Input, Hidden, Output).

- Lernfähig: Implementierung von Backpropagation und Gradient Descent (Batch/Stochastic).

- Aktivierungsfunktionen: Unterstützt Identität, Step und Sigmoid (inkl. Ableitungen).

- Daten-Handling:

  - Automatischer Download und Parsing der MNIST-Dateien (`.gz`).

  - Konvertierung eigener `.png` Bilder in normalisierte Arrays.

- Beispiele:

  - Klassifikation (Ziffernerkennung 0-9).

  - Regression (Mietpreisvorhersage im `TestDatenSatz`).

## Projektstruktur

Der Quellcode befindet sich im Ordner `src`:

- `Main.java`: Der Haupteinstiegspunkt.

  - Lädt die MNIST-Daten (Training & Test).

  - Initialisiert das Netz (784 Input -> 128 Hidden -> 10 Output).

  - Führt das Training über mehrere Epochen durch.

  - Testet die Genauigkeit und kann optional eigene Bilder aus einem data/ Ordner vorhersagen.

- `Netz.java`: Die Kernklasse. Verwaltet die Schichten (`Schicht`) und Neuronen. Führt `forwardPass` (Vorhersage) und `backwardPass` (Lernen/Gewichtsanpassung) aus.

- `Schicht.java` & `Neuron.java`: Repräsentieren die Netzarchitektur. Neuronen speichern Gewichte, Bias und führen die Aktivierungsfunktionen aus.

- `MnistLoader.java`: Utility-Klasse zum Herunterladen und Einlesen der `idx1-ubyte` und `idx3-ubyte` Formate des MNIST-Datensatzes.

- `ActFuntions.java`: Sammlung mathematischer Funktionen (Sigmoid, Step, etc.) und deren Ableitungen für den Lernprozess.

- `PNGArr.java`: Hilfsklasse zum Einlesen von Bildern und Umwandeln in Graustufen-Arrays (0.0 - 1.0).

- `TestDatenSatz.java`: Ein separates, kleineres Beispiel für Regressionsaufgaben (Wohnungsmiete basierend auf Größe und Zimmern).

## Schnellstart

### Voraussetzungen

- Java Development Kit (JDK) installiert (empfohlen JDK 17 oder neuer, Projekt nutzt aktuell Level 23).

- Eine IDE wie IntelliJ IDEA oder Eclipse (optional, aber empfohlen).

### Ausführung (MNIST Ziffernerkennung)

1. Öffne das Projekt in deiner IDE.

2. Führe die Klasse `Main.java` aus.

3. **Automatischer Download:** Beim ersten Start lädt das Programm die MNIST-Daten automatisch in den Ordner `mnist/` herunter.

4. **Training:** Das Netz trainiert (standardmäßig 2 Epochen) und gibt die Genauigkeit (Accuracy) und den Fehler (MSE) in der Konsole aus.

5. **Test:** Abschließend wird das Netz gegen den Test-Datensatz validiert.

### Eigene Bilder testen

Um eigene Handschriften zu testen:

1. Erstelle einen Ordner `data` im Projektverzeichnis (auf gleicher Ebene wie src).

2. Lege `.png` Dateien darin ab (z.B. schwarze Schrift auf weißem Grund oder umgekehrt, das Skript normalisiert dies).

3. Beim erneuten Ausführen von `Main` werden diese Bilder automatisch eingelesen, auf 28x28 Pixel skaliert und klassifiziert.

### Kleines Regressions-Beispiel

Um zu sehen, wie das Netz einfache Zusammenhänge lernt (z.B. Miete berechnen):

- Führe `TestDatenSatz.java` aus. Dies trainiert ein kleines Netz auf wenigen Beispieldaten.

## Hinweise zur Konfiguration

In der Main.java können Parameter angepasst werden:

- Lernrate: netz.setLearningRate(0.01) – Beeinflusst, wie schnell das Netz lernt.

- Epochen: `int epochs = 2` – Anzahl der Durchläufe durch den gesamten Datensatz.

- Netzarchitektur: `new Netz(128, 10)` – Anzahl der Neuronen in den Hidden- und Output-Layern (Input ergibt sich aus den Daten).

## **Bekannte Limitierungen**

- Das Training läuft Single-Threaded auf der CPU.

- Keine Speicherung (Persistierung) des trainierten Modells (Gewichte gehen nach Programmende verloren).