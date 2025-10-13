import java.util.ArrayList;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class testen {
    @Test
    public void autoTest() {
        Netz netz = new Netz(4, 1);
        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                netz.setNeuronFkt(1, i, 0);
                netz.setNeuronFkt(2, i, 1);
            } else {
                netz.setNeuronFkt(1, i, 1);
            }
        }
        netz.setNeuronWeights(2, 1, 1, 0.5);
        netz.setNeuronWeights(2, 1, 2, 0.2);
        netz.setNeuronWeights(2, 1, 3, 0.6);
        netz.setNeuronWeights(2, 1, 4, 0.1);
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(0.2);
        eingabe.add(0.8);
        eingabe.add(0.6);
        eingabe.add(1.0);
        double d = netz.vorwaerts(eingabe);
        assertEquals(0.87, d);

    }

    @Test
    public void gruppe2() {
        Netz netz = new Netz(3, 3, 2);
        netz.setNeuronFkt(1, 1, 0);
        netz.setNeuronFkt(1, 2, 0);
        netz.setNeuronFkt(1, 3, 0);
        netz.setNeuronFkt(2, 1, 0);
        netz.setNeuronFkt(2, 2, 0);
        netz.setNeuronFkt(2, 3, 0);
        netz.setNeuronFkt(3, 1, 0);
        netz.setNeuronFkt(3, 2, 0);

        netz.setNeuronWeights(2, 1, 1, -0.5);
        netz.setNeuronWeights(2, 1, 2, -0.4);
        netz.setNeuronWeights(2, 1, 3, -0.3);
        netz.setNeuronWeights(2, 2, 1, -0.2);
        netz.setNeuronWeights(2, 2, 2, -0.1);
        netz.setNeuronWeights(2, 2, 3, 0.1);
        netz.setNeuronWeights(3, 1, 1, 0.1);
        netz.setNeuronWeights(3, 1, 2, 0.2);
        netz.setNeuronWeights(3, 1, 3, 0.3);
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(-10.0);
        eingabe.add(10.0);
        assertEquals(3.02, netz.vorwaerts(eingabe));


    }
}