import java.util.ArrayList;
import java.util.Scanner;

public class Netz {
    // Liste der Schichten des Netzes
    private int firstLayerNeurons;
    private ArrayList<Schicht> schichten = new ArrayList<>();
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
    public double vorwaerts() {
        ArrayList<Double> aktuelleEingabe = new ArrayList<>(input);

        for (Schicht schicht : schichten) {
            double summe = schicht.schichtSum(aktuelleEingabe);
            ArrayList<Double> naechsteEingabe = new ArrayList<>(1);
            naechsteEingabe.add(summe);
            aktuelleEingabe = naechsteEingabe;
        }
        return aktuelleEingabe.getFirst();
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
}
