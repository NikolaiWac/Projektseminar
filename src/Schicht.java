import java.util.ArrayList;

public class Schicht {
    ArrayList<Neuron> neuronen;

    public Neuron getNeuron(int pos) {
        return neuronen.get(pos);
    }

    public ArrayList<Neuron> getNeuronen() {
        return neuronen;
    }

    //Initialisiert die gewünschte anzahl an Neuronen
    //Anzahl kommt aus input Netz-Klasse
    public Schicht(int anzNeuron, int inputsCount) {
        neuronen = new ArrayList<>();
        for (int i = 0; i < anzNeuron; i++) {
            neuronen.add(new Neuron(0, inputsCount));
        }
    }

    //Berechnet die Summe aller Neuronen in der Schicht
    //Benötigt summe von vorheriger schicht aus Netz-Klasse
    public ArrayList<Double> schichtSum(ArrayList<Double> input, double bias) {
        ArrayList<Double> sum = new ArrayList<>();
        for (int i = 0; i < neuronen.size(); i++) {
            sum.add(neuronen.get(i).outputFkt(input, bias));
        }
        return sum;
    }
}
