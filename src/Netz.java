
import java.util.ArrayList;
import java.util.stream.Collectors;

public class Netz {
    private double bias = 0;
    private int firstLayerNeurons;
    // Liste der Schichten des Netzes
    private ArrayList<Schicht> schichten = new ArrayList<>();
    // Hauptinput des Netzes
    private ArrayList<Double> input;

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
        ArrayList<Double> outputDeltas = new ArrayList<>();
        ArrayList<Neuron> currentLayer = schichten.getLast().getNeuronen();
        outputDeltas = currentLayer.stream().map(e -> (ActFuntions.derivativeSelect(e.aktFkt, e.getIn()) * (expectedValue - e.getOut()))).collect(Collectors.toCollection(ArrayList::new));
        deltas.add(outputDeltas);
        //Durchlaufen für jede Schicht
        for (int i = schichten.size() - 2; i >= 0; i--) {
            currentLayer = schichten.get(i).getNeuronen();
            ArrayList<Neuron> nextLayer = schichten.get(i + 1).getNeuronen();
            ArrayList<Double> prevDeltas = deltas.get(0);
            ArrayList<Double> currentDeltas = new ArrayList<>();
            for (int j = 0; j < nextLayer.size(); j++) {
                Neuron neuron = currentLayer.get(j);
                double sum = 0.0;

                // Summe der gewichteten Deltas der nächsten Schicht
                for (int k = 0; k < schichten.size(); k++) {
                    sum += nextLayer.get(k).getWeights().get(j) * prevDeltas.get(k);
                }
                double delta = ActFuntions.derivativeSelect(neuron.aktFkt, neuron.getIn()) * sum;
                currentDeltas.add(delta);
            }
            // Deltas vorne einfügen (damit Indexierung stimmt)
            deltas.addFirst(currentDeltas);
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
