
import java.util.ArrayList;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Netz {
    ArrayList<ArrayList<Double>> forwardPassResults;
    ArrayList<ArrayList<Double>> neuronInputs;
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
            aktuelleEingabe = schicht.schichtSum(aktuelleEingabe, bias).stream().map(e -> e[0]).collect(Collectors.toCollection(ArrayList::new));
            neuronInputs.add(schicht.schichtSum(aktuelleEingabe, bias).stream().map(e -> e[1]).collect(Collectors.toCollection(ArrayList::new)));
            forwardPassResults = new ArrayList<>();
            forwardPassResults.add(aktuelleEingabe);
        }
        return aktuelleEingabe.stream().mapToDouble(Double::doubleValue).sum();
    }

    public void backwardPass(Double expectedValue){
        ArrayList<ArrayList<Double>> deltas = new ArrayList<>();
        ArrayList<Double> outputDeltas = new ArrayList<>();
        for(int i = 0; i < forwardPassResults.getLast().size(); i++){
            double error = (expectedValue - neuronInputs.getLast().get(i));
            double delta = Aktivierungsfunktionen.derivativeSelect(getNeuron(forwardPassResults.size()-1, i).aktFkt, neuronInputs.getLast().get(i)) * error;
            outputDeltas.add(delta);
        }
        deltas.add(outputDeltas);
        for(int i = schichten.size()-2; i >= 0; i--){
            double delta = Aktivierungsfunktionen.derivativeSelect(getNeuron(i, ).aktFkt, neuronInputs.getLast().get(i))

        }
    }

    /*public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Anzahl der Schichten eingeben (>= 2):");
        int anzahlSchichten = readPositiveInt(sc);

        int[] neuronenProSchicht = new int[anzahlSchichten];
        for (int i = 0; i < anzahlSchichten; i++) {
            System.out.println("Anzahl der Neuronen für Schicht " + (i + 1) + " eingeben (>= 1):");
            neuronenProSchicht[i] = readPositiveInt(sc);
        }

        Netz netz = new Netz(neuronenProSchicht);

        System.out.println("Anzahl der Eingabewerte eingeben (>= 1)");
        int anzahlInputs = readNonNegativeInt(sc);
        ArrayList<Double> input = new ArrayList<>();
        for (int i = 0; i < anzahlInputs; i++) {
            System.out.println("Eingabewert " + (i + 1) + ":");
            input.add(readDouble(sc));
        }
        double output = netz.vorwaerts(input);
        System.out.println("Netz-Output: " + output);

    }*/

    private static int readPositiveInt(Scanner sc) {
        while (true) {
            try {
                int v = Integer.parseInt(sc.nextLine().trim());
                if (v > 0) return v;
            } catch (Exception ignored) {
            }
            System.out.println("Bitte eine ganze Zahl > 0 eingeben:");
        }
    }

    private static int readNonNegativeInt(Scanner sc) {
        while (true) {
            try {
                int v = Integer.parseInt(sc.nextLine().trim());
                if (v >= 0) return v;
            } catch (Exception ignored) {
            }
            System.out.println("Bitte eine ganze Zahl >= 0 eingeben:");
        }
    }

    private static double readDouble(Scanner sc) {
        while (true) {
            try {
                return Double.parseDouble(sc.nextLine().trim());
            } catch (Exception ignored) {
                System.out.println("Bitte eine gültige Zahl eingeben:");
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
