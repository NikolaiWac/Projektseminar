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

    // Compute outputs into a provided preallocated array (parallel)
    public void computeIntoParallel(double[] input, double bias, double[] out) {
        Neuron[] local = this.neuronen;
        IntStream.range(0, local.length).parallel().forEach(i -> {
            out[i] = local[i].outputFkt(input, bias);
        });
    }
}
