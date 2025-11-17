
import java.util.ArrayList;
import java.util.Iterator;
import java.util.stream.Collectors;

public class Netz {
    private double bias = 0;
    private int firstLayerNeurons;
    // Liste der Schichten des Netzes
    private ArrayList<Schicht> schichten = new ArrayList<>();
    // Hauptinput des Netzes
    private ArrayList<Double> input;
    private double learningRate = 1.0;

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    // Konstruktor: Erstellt für jede übergebene Zahl eine Schicht außer für erste Schicht
    // mit entsprechend vielen Neuronen
    // aktuell alle Schichten erstellt bis auf die erste, Neuronen in der jeweiligen Schicht kriegen direkt anzahl an inputs
    public Netz(int... anzahlNeuronenProSchicht) {
        if (anzahlNeuronenProSchicht != null) {
            firstLayerNeurons = anzahlNeuronenProSchicht[0];
            for (int i = 1; i < anzahlNeuronenProSchicht.length; i++) {
                schichten.add(new Schicht(anzahlNeuronenProSchicht[i], anzahlNeuronenProSchicht[i - 1]));
            }
        }
    }

    //hier wird der input fürs netz übergeben, erst jetzt kann das netzt genutzt werden
    //erste Schicht wird erst hier erzeugt, denn erst hier klar wieviele inputs die erste Schicht bekommt
    public void init(ArrayList<Double> input) {
        if (input != null) {
            this.input = input;
            schichten.addFirst(new Schicht(firstLayerNeurons, input.size()));

        }
    }

    // Durchläuft das Netzwerk: Input -> Eingaben als Liste
    // Output -> Ausgabe als Double
    public double forwardPass() {
        ArrayList<Double> aktuelleEingabe = new ArrayList<>(input);
        for (Schicht schicht : schichten) {
            aktuelleEingabe = schicht.schichtSum(aktuelleEingabe, bias);
        }
        return aktuelleEingabe.stream().mapToDouble(Double::doubleValue).sum();
    }

    public void backwardPass(Double expectedValue) {
        forwardPass();
        ArrayList<ArrayList<Double>> deltas = new ArrayList<>();
        ArrayList<Neuron> currentLayer = schichten.getLast().getNeuronen();
        ArrayList<Double> outputDeltas = currentLayer.stream().map(e -> (ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (expectedValue - e.getOut()))).collect(Collectors.toCollection(ArrayList::new));
        deltas.add(outputDeltas);
        //Durchlaufen für jede Schicht
        for (int i = schichten.size() - 2; i >= 0; i--) {
            currentLayer = schichten.get(i).getNeuronen();
            ArrayList<Neuron> nextLayer = schichten.get(i + 1).getNeuronen();
            ArrayList<Double> prevDeltas = deltas.getFirst();
            ArrayList<Double> currentDeltas = new ArrayList<>();
            //Iterieren über die Neuronen der aktuellen Schicht, und berechen für jedes den Fehler/Delta
            for (int j = 0; j < currentLayer.size(); j++) {
                Neuron neuron = currentLayer.get(j);
                double sum = 0.0;
                // Iterieren über die vorherige Schicht(beim Folienbeispiel Schicht K)
                // Multipliziert dann das Gewicht zwischen aktuellen Neuron J und vorherigen Neuron K mit dem Fehler vom Neuron K
                for (int k = 0; k < nextLayer.size(); k++) {
                    sum += nextLayer.get(k).getWeights().get(j) * prevDeltas.get(k);
                }
                double delta = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                currentDeltas.add(delta);
            }
            // Deltas vorne einfügen, damit deltas.get(0) funktioniert
            deltas.addFirst(currentDeltas);
        }
        //setWeights output Layer
        for (int j = 0; j < schichten.getLast().getNeuronen().size(); j++) {
            Neuron n = schichten.getLast().getNeuronen().get(j);
            for (int i = 0; i < n.getWeights().size(); i++) {
                int prevLayer = schichten.size() - 2;
                double delta = deltas.getLast().get(j);
                double newWeight = n.getWeights().get(i) + (learningRate * getNeuron(prevLayer, i).getOut() * delta);
                n.setWeights(i, newWeight);
            }
        }
        //setWeights hiddenLayers
        for (int j = schichten.size() - 2; j > 0 ; j--) {
            for (int k = 0; k < schichten.get(j).getNeuronen().size(); k++) {
                Neuron n = schichten.get(j).getNeuronen().get(k);
                for (int i = 0; i < n.getWeights().size(); i++) {
                    double newWeight = n.getWeights().get(i) + (learningRate * getNeuron(j-1, i).getOut() * deltas.get(j).get(k));
                    n.setWeights(i, newWeight);
                }
            }
        }
        //setWeights input Layer
        for (int j = 0; j < schichten.getFirst().getNeuronen().size(); j++) {
            Neuron n = schichten.getFirst().getNeuronen().get(j);
            for (int i = 0; i < n.getWeights().size(); i++) {
                double delta = deltas.getFirst().get(j);
                double newWeight = n.getWeights().get(i) + (learningRate * input.get(i) * delta);
                n.setWeights(i, newWeight);
            }
        }
    }

    //setNeuron Funktionen aktuell ohne Fehlermeldungen
    public void setNeuronFkt(int layer, int pos, int fkt) {
        schichten.get(layer).getNeuron(pos).setAktFkt(fkt);
    }

    public void setNeuronFkt(int layer, int pos, int fkt, ArrayList<Double> furtherInfo) {
        schichten.get(layer).getNeuron(pos).setAktFkt(fkt, furtherInfo);
    }


    public void setNeuronWeights(int layer, int pos, int inputPos, double weight) {
        schichten.get(layer).getNeuron(pos).setWeights(inputPos, weight);
    }

    public Neuron getNeuron(int layer, int pos) {
        return schichten.get(layer).getNeuron(pos);
    }

    public void setBiasWeights(int layer, int pos, double weight) {
        schichten.get(layer).getNeuron(pos).setBiasWeight(weight);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
