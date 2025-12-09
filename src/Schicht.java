import java.util.stream.IntStream;

public class Schicht {
    Neuron[] neuronen;

    public Neuron getNeuron(int pos) {
        return neuronen[pos];
    }

    public Neuron[] getNeuronen() {
        return neuronen;
    }

    //Initialisiert die gewünschte anzahl an Neuronen
    //Anzahl kommt aus input Netz-Klasse
    public Schicht(int anzNeuron, int inputsCount) {
        neuronen = new Neuron[anzNeuron];
        for (int i = 0; i < anzNeuron; i++) {
            neuronen[i] = new Neuron(0, inputsCount, i);
        }
    }

    //Berechnet die Summe aller Neuronen in der Schicht
    //Benötigt summe von vorheriger schicht aus Netz-Klasse
    public double[] schichtSum(double[] input, double bias) {
        double[] sum = new double[neuronen.length];
        for (int i = 0; i < neuronen.length; i++) {
            sum[i] = neuronen[i].outputFkt(input, bias);
        }
        return sum;
    }

    // Parallele Variante für Multi-Core Nutzung
    public double[] schichtSumParallel(double[] input, double bias) {
        double[] sum = new double[neuronen.length];
        IntStream.range(0, neuronen.length).parallel().forEach(i -> {
            sum[i] = neuronen[i].outputFkt(input, bias);
        });
        return sum;
    }
}
