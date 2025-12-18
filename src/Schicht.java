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
        double[] out = new double[neuronen.length];
        computeInto(input, bias, out);
        return out;
    }

    // Parallele Variante für Multi-Core Nutzung
    public double[] schichtSumParallel(double[] input, double bias) {
        double[] out = new double[neuronen.length];
        computeIntoParallel(input, bias, out);
        return out;
    }

    // Compute outputs into a provided preallocated array (sequential)
    public void computeInto(double[] input, double bias, double[] out) {
        // assume caller ensures out.length == neuronen.length
        Neuron[] local = this.neuronen;
        for (int i = 0; i < local.length; i++) {
            out[i] = local[i].outputFkt(input, bias);
        }
    }

    // Compute outputs into a provided preallocated array (parallel)
    public void computeIntoParallel(double[] input, double bias, double[] out) {
        Neuron[] local = this.neuronen;
        IntStream.range(0, local.length).parallel().forEach(i -> {
            out[i] = local[i].outputFkt(input, bias);
        });
    }
}
