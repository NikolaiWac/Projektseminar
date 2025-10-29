import java.util.ArrayList;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class testen {
    @Test
    public void autoTest() {
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(0.2);
        eingabe.add(0.8);
        eingabe.add(0.6);
        eingabe.add(1.0);
        Netz netz = new Netz(4, 1);
        netz.init(eingabe);

        ArrayList<Double> furtherInfo = new ArrayList<>(){
            {
                add(1.0);
                add(0.0);
                add(1.0);
            }
        };

        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                netz.setNeuronFkt(0, i, 0);
                netz.setNeuronFkt(1, i, 1, furtherInfo);
            } else {
                netz.setNeuronFkt(0, i, 1, furtherInfo);
            }
        }

        netz.setNeuronWeights(1, 0, 0, 0.5);
        netz.setNeuronWeights(1, 0, 1, 0.2);
        netz.setNeuronWeights(1, 0, 2, 0.6);
        netz.setNeuronWeights(1, 0, 3, 0.1);

        double d = netz.vorwaerts();
        assertEquals(0.87, d);

    }

    @Test
    public void gruppe2() {
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(-10.0);
        eingabe.add(10.0);
        Netz netz = new Netz(3, 3, 2);
        netz.init(eingabe);

        netz.setNeuronFkt(0, 0, 0);
        netz.setNeuronFkt(0, 1, 0);
        netz.setNeuronFkt(0, 2, 0);
        netz.setNeuronFkt(1, 0, 0);
        netz.setNeuronFkt(1, 1, 0);
        netz.setNeuronFkt(1, 2, 0);
        netz.setNeuronFkt(2, 0, 0);
        netz.setNeuronFkt(2, 1, 0);

        netz.setNeuronWeights(1, 0, 0, -0.5);
        netz.setNeuronWeights(1, 0, 1, -0.4);
        netz.setNeuronWeights(1, 0, 2, -0.3);
        netz.setNeuronWeights(1, 1, 0, -0.2);
        netz.setNeuronWeights(1, 1, 1, -0.1);
        netz.setNeuronWeights(1, 1, 2, 0.1);
        netz.setNeuronWeights(2, 0, 0, 0.1);
        netz.setNeuronWeights(2, 0, 1, 0.2);
        netz.setNeuronWeights(2, 0, 2, 0.3);

        assertEquals(3.02, netz.vorwaerts());
    }
}