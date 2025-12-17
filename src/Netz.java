import java.util.stream.IntStream;

public class Netz {
    private double bias = 0;
    private int firstLayerNeurons;
    // Schichten des Netzes
    private Schicht[] schichten;
    // Hauptinput des Netzes
    private double[] input;
    private double learningRate = 1.0;
    private boolean parallelEnabled = false;
    private int[] neuronsPerLayer;

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public boolean isParallelEnabled() {
        return parallelEnabled;
    }

    public void setParallelEnabled(boolean enabled) {
        this.parallelEnabled = enabled;
    }

    // Konstruktor: Erstellt für jede übergebene Zahl eine Schicht außer für erste Schicht
    // mit entsprechend vielen Neuronen
    // aktuell alle Schichten erstellt bis auf die erste, Neuronen in der jeweiligen Schicht kriegen direkt anzahl an inputs
    public Netz(int... anzahlNeuronenProSchicht) {
        if (anzahlNeuronenProSchicht != null && anzahlNeuronenProSchicht.length > 0) {
            this.neuronsPerLayer = anzahlNeuronenProSchicht.clone();
            this.firstLayerNeurons = anzahlNeuronenProSchicht[0];
            this.schichten = new Schicht[anzahlNeuronenProSchicht.length];
        }
    }

    //hier wird der input fürs netz übergeben, erst jetzt kann das netzt genutzt werden
    //erste Schicht wird erst hier erzeugt, denn erst hier klar wieviele inputs die erste Schicht bekommt
    public void init(double[] input) {
        if (input != null) {
            this.input = input;
            // Erste Schicht kennt erst jetzt die Anzahl an Inputs
            schichten[0] = new Schicht(firstLayerNeurons, input.length);
            // Restliche Schichten hängen nur von Anzahl Neuronen der vorherigen Schicht ab
            for (int i = 1; i < schichten.length; i++) {
                schichten[i] = new Schicht(neuronsPerLayer[i], neuronsPerLayer[i - 1]);
            }
        }
    }

    // Durchläuft das Netzwerk: Input -> Eingaben als Liste
    // Output -> Ausgabe als Double
    public double forwardPass() {
        double[] aktuelleEingabe = input.clone();
        for (int i = 0; i < schichten.length; i++) {
            aktuelleEingabe = parallelEnabled
                    ? schichten[i].schichtSumParallel(aktuelleEingabe, bias)
                    : schichten[i].schichtSum(aktuelleEingabe, bias);
        }
        double sum = 0.0;
        for (double v : aktuelleEingabe) sum += v;
        return sum;
    }

    // Forward pass that returns the full output vector of the last layer
    public double[] forwardPassVector() {
        double[] aktuelleEingabe = input.clone();
        for (int i = 0; i < schichten.length; i++) {
            aktuelleEingabe = parallelEnabled
                    ? schichten[i].schichtSumParallel(aktuelleEingabe, bias)
                    : schichten[i].schichtSum(aktuelleEingabe, bias);
        }
        return aktuelleEingabe; // outputs of the last layer
    }

    public void backwardPass(double expectedValue) {
        // Führt Forward Pass aus, um In/Out in Neuronen zu aktualisieren
        forwardPass();

        int L = schichten.length;
        // Deltas pro Schicht, gleiche Indizierung wie schichten
        double[][] deltas = new double[L][];

        // Output-Layer Deltas
        int last = L - 1;
        Neuron[] outLayer = schichten[last].getNeuronen();
        deltas[last] = new double[outLayer.length];
        if (parallelEnabled) {
            IntStream.range(0, outLayer.length).parallel().forEach(j -> {
                Neuron e = outLayer[j];
                deltas[last][j] = ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (expectedValue - e.getOut());
            });
        } else {
            for (int j = 0; j < outLayer.length; j++) {
                Neuron e = outLayer[j];
                deltas[last][j] = ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (expectedValue - e.getOut());
            }
        }

        // Hidden-Layer Deltas rückwärts berechnen
        for (int i = L - 2; i >= 0; i--) {
            Neuron[] currentLayer = schichten[i].getNeuronen();
            Neuron[] nextLayer = schichten[i + 1].getNeuronen();
            deltas[i] = new double[currentLayer.length];
            if (parallelEnabled) {
                final int iLayer = i;
                final Neuron[] nextLayerLocal = nextLayer;
                IntStream.range(0, currentLayer.length).parallel().forEach(j -> {
                    Neuron neuron = currentLayer[j];
                    double sum = 0.0;
                    for (int k = 0; k < nextLayerLocal.length; k++) {
                        sum += nextLayerLocal[k].getWeights()[j] * deltas[iLayer + 1][k];
                    }
                    deltas[iLayer][j] = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                });
            } else {
                for (int j = 0; j < currentLayer.length; j++) {
                    Neuron neuron = currentLayer[j];
                    double sum = 0.0;
                    for (int k = 0; k < nextLayer.length; k++) {
                        sum += nextLayer[k].getWeights()[j] * deltas[i + 1][k];
                    }
                    deltas[i][j] = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                }
            }
        }

        // Gewichte Output-Layer anpassen
        int prevLayerIdx = L - 2;
        if (parallelEnabled) {
            IntStream.range(0, outLayer.length).parallel().forEach(j -> {
                Neuron n = outLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[last][j];
                    double newWeight = w[i2] + (learningRate * getNeuron(prevLayerIdx, i2).getOut() * delta);
                    n.setWeights(i2, newWeight);
                }
            });
        } else {
            for (int j = 0; j < outLayer.length; j++) {
                Neuron n = outLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[last][j];
                    double newWeight = w[i2] + (learningRate * getNeuron(prevLayerIdx, i2).getOut() * delta);
                    n.setWeights(i2, newWeight);
                }
            }
        }

        // Gewichte Hidden-Layer anpassen (ohne Input-Layer)
        for (int layer = L - 2; layer > 0; layer--) {
            Neuron[] layerNeurons = schichten[layer].getNeuronen();
            if (parallelEnabled) {
                final int layerIdx = layer;
                IntStream.range(0, layerNeurons.length).parallel().forEach(k -> {
                    Neuron n = layerNeurons[k];
                    double[] w = n.getWeights();
                    for (int i2 = 0; i2 < w.length; i2++) {
                        double newWeight = w[i2] + (learningRate * getNeuron(layerIdx - 1, i2).getOut() * deltas[layerIdx][k]);
                        n.setWeights(i2, newWeight);
                    }
                });
            } else {
                for (int k = 0; k < layerNeurons.length; k++) {
                    Neuron n = layerNeurons[k];
                    double[] w = n.getWeights();
                    for (int i2 = 0; i2 < w.length; i2++) {
                        double newWeight = w[i2] + (learningRate * getNeuron(layer - 1, i2).getOut() * deltas[layer][k]);
                        n.setWeights(i2, newWeight);
                    }
                }
            }
        }

        // Gewichte Input-Layer anpassen (Layer 0 nutzt direkt die Eingangswerte)
        Neuron[] firstLayer = schichten[0].getNeuronen();
        if (parallelEnabled) {
            IntStream.range(0, firstLayer.length).parallel().forEach(j -> {
                Neuron n = firstLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[0][j];
                    double newWeight = w[i2] + (learningRate * input[i2] * delta);
                    n.setWeights(i2, newWeight);
                }
            });
        } else {
            for (int j = 0; j < firstLayer.length; j++) {
                Neuron n = firstLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[0][j];
                    double newWeight = w[i2] + (learningRate * input[i2] * delta);
                    n.setWeights(i2, newWeight);
                }
            }
        }
    }

    // Backpropagation for vector outputs (e.g., multi-class). expected must have the same
    // length as the number of neurons in the output layer.
    public void backwardPassVector(double[] expected) {
        // Run forward pass to ensure neuron in/out are up-to-date
        forwardPass();

        int L = schichten.length;
        double[][] deltas = new double[L][];

        // Output layer deltas (per neuron target)
        int last = L - 1;
        Neuron[] outLayer = schichten[last].getNeuronen();
        deltas[last] = new double[outLayer.length];
        if (parallelEnabled) {
            IntStream.range(0, outLayer.length).parallel().forEach(j -> {
                Neuron e = outLayer[j];
                double target = (expected != null && j < expected.length) ? expected[j] : 0.0;
                deltas[last][j] = ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (target - e.getOut());
            });
        } else {
            for (int j = 0; j < outLayer.length; j++) {
                Neuron e = outLayer[j];
                double target = (expected != null && j < expected.length) ? expected[j] : 0.0;
                deltas[last][j] = ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (target - e.getOut());
            }
        }

        // Hidden layers deltas (backwards)
        for (int i = L - 2; i >= 0; i--) {
            Neuron[] currentLayer = schichten[i].getNeuronen();
            Neuron[] nextLayer = schichten[i + 1].getNeuronen();
            deltas[i] = new double[currentLayer.length];
            if (parallelEnabled) {
                final int iLayer = i;
                final Neuron[] nextLayerLocal = nextLayer;
                IntStream.range(0, currentLayer.length).parallel().forEach(j -> {
                    Neuron neuron = currentLayer[j];
                    double sum = 0.0;
                    for (int k = 0; k < nextLayerLocal.length; k++) {
                        sum += nextLayerLocal[k].getWeights()[j] * deltas[iLayer + 1][k];
                    }
                    deltas[iLayer][j] = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                });
            } else {
                for (int j = 0; j < currentLayer.length; j++) {
                    Neuron neuron = currentLayer[j];
                    double sum = 0.0;
                    for (int k = 0; k < nextLayer.length; k++) {
                        sum += nextLayer[k].getWeights()[j] * deltas[i + 1][k];
                    }
                    deltas[i][j] = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                }
            }
        }

        // Update weights for output layer
        int prevLayerIdx = L - 2;
        if (parallelEnabled) {
            IntStream.range(0, outLayer.length).parallel().forEach(j -> {
                Neuron n = outLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[last][j];
                    double newWeight = w[i2] + (learningRate * getNeuron(prevLayerIdx, i2).getOut() * delta);
                    n.setWeights(i2, newWeight);
                }
            });
        } else {
            for (int j = 0; j < outLayer.length; j++) {
                Neuron n = outLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[last][j];
                    double newWeight = w[i2] + (learningRate * getNeuron(prevLayerIdx, i2).getOut() * delta);
                    n.setWeights(i2, newWeight);
                }
            }
        }

        // Update weights for hidden layers (excluding input layer)
        for (int layer = L - 2; layer > 0; layer--) {
            Neuron[] layerNeurons = schichten[layer].getNeuronen();
            if (parallelEnabled) {
                final int layerIdx = layer;
                IntStream.range(0, layerNeurons.length).parallel().forEach(k -> {
                    Neuron n = layerNeurons[k];
                    double[] w = n.getWeights();
                    for (int i2 = 0; i2 < w.length; i2++) {
                        double newWeight = w[i2] + (learningRate * getNeuron(layerIdx - 1, i2).getOut() * deltas[layerIdx][k]);
                        n.setWeights(i2, newWeight);
                    }
                });
            } else {
                for (int k = 0; k < layerNeurons.length; k++) {
                    Neuron n = layerNeurons[k];
                    double[] w = n.getWeights();
                    for (int i2 = 0; i2 < w.length; i2++) {
                        double newWeight = w[i2] + (learningRate * getNeuron(layer - 1, i2).getOut() * deltas[layer][k]);
                        n.setWeights(i2, newWeight);
                    }
                }
            }
        }

        // Update weights for input layer (layer 0 uses raw inputs)
        Neuron[] firstLayer = schichten[0].getNeuronen();
        if (parallelEnabled) {
            IntStream.range(0, firstLayer.length).parallel().forEach(j -> {
                Neuron n = firstLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[0][j];
                    double newWeight = w[i2] + (learningRate * input[i2] * delta);
                    n.setWeights(i2, newWeight);
                }
            });
        } else {
            for (int j = 0; j < firstLayer.length; j++) {
                Neuron n = firstLayer[j];
                double[] w = n.getWeights();
                for (int i2 = 0; i2 < w.length; i2++) {
                    double delta = deltas[0][j];
                    double newWeight = w[i2] + (learningRate * input[i2] * delta);
                    n.setWeights(i2, newWeight);
                }
            }
        }
    }

    //setNeuron Funktionen aktuell ohne Fehlermeldungen
    public void setNeuronFkt(int layer, int pos, int fkt) {
        schichten[layer].getNeuron(pos).setAktFkt(fkt);
    }

    public void setNeuronFkt(int layer, int pos, int fkt, double[] furtherInfo) {
        schichten[layer].getNeuron(pos).setAktFkt(fkt, furtherInfo);
    }


    public void setNeuronWeights(int layer, int pos, int inputPos, double weight) {
        schichten[layer].getNeuron(pos).setWeights(inputPos, weight);
    }

    public Neuron getNeuron(int layer, int pos) {
        return schichten[layer].getNeuron(pos);
    }

    public void setBiasWeights(int layer, int pos, double weight) {
        schichten[layer].getNeuron(pos).setBiasWeight(weight);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
