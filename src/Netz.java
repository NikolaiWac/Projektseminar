import java.util.ArrayList;
import java.util.Scanner;

public class Netz {
    // Liste der Schichten des Netzes
    private final ArrayList<Schicht> schichten = new ArrayList<>();

    // Konstruktor: Erstellt für jede übergebene Zahl eine Schicht
    // mit entsprechend vielen Neuronen
    public Netz(int... anzahlNeuronenProSchicht) {
        if (anzahlNeuronenProSchicht != null) {
            for (int anz : anzahlNeuronenProSchicht) {
                schichten.add(new Schicht(anz));
            }
        }
    }

    // Durchläuft das Netzwerk: Input -> Eingaben als Liste
    // Output -> Ausgabe als Double
    public double vorwaerts(ArrayList<Double> eingabe) {
        if (eingabe == null) {
            throw new IllegalArgumentException("Eingabe darf nicht null sein");
        }
        ArrayList<Double> aktuelleEingabe = new ArrayList<>(eingabe);

        for (Schicht schicht : schichten) {
            double summe = schicht.schichtSum(aktuelleEingabe);
            ArrayList<Double> naechsteEingabe = new ArrayList<>(1);
            naechsteEingabe.add(summe);
            aktuelleEingabe = naechsteEingabe;
        }
        return aktuelleEingabe.getFirst();
    }

    public static void main(String[] args) {
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

    }

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

    public void setNeuronWeights(int layer, int pos, int inputPos, double weight) {
        schichten.get(layer).getNeuron(pos).setWeights(inputPos, weight);
    }
}
