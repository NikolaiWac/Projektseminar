import java.util.ArrayList;

public class Schicht {
    ArrayList<Neuron> neuronen;

    //Initialisiert die gewünschte anzahl an Neuronen
    //Anzahl kommt aus input Netz-Klasse
    public Schicht(int anzNeuron) {
        neuronen = new ArrayList<>();
        for (int i = 0; i < anzNeuron; i++) {
            neuronen.add(new Neuron(0));
        }
    }
    //Berechnet die Summe aller Neuronen in der Schicht
    //Benötigt summe von vorheriger schicht aus Netz-Klasse
    public double schichtSum(ArrayList<Double> input){
        double sum = 0;
        for (int i = 0; i < neuronen.size(); i++) {
            sum += neuronen.get(i).outputFkt(input);
        }
        return sum;
    }
}
