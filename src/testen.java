import java.util.ArrayList;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class testen {
    @Test
    public void test1() {
        //erstellen des Hauptinputs für das Netz
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(0.2);
        eingabe.add(0.8);
        eingabe.add(0.6);
        eingabe.add(1.0);
        Netz netz = new Netz( 1);
        netz.init(eingabe);

        //Informationen für die Aktivierungsfunktion
        ArrayList<Double> furtherInfo = new ArrayList<>(){
            {
                add(1.0);
                add(0.0);
                add(1.0);
            }
        };
        netz.setNeuronFkt(0, 0, 1, furtherInfo);

        netz.setNeuronWeights(0, 0, 0, 0.5);
        netz.setNeuronWeights(0, 0, 1, 0.2);
        netz.setNeuronWeights(0, 0, 2, 0.6);
        netz.setNeuronWeights(0, 0, 3, 0.1);

        double d = netz.forwardPass();
        assertEquals(0.0, d);

    }

    @Test
    public void test2() {
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(-10.0);
        eingabe.add(10.0);
        Netz netz = new Netz( 2, 2);
        netz.init(eingabe);
        netz.setBias(1.0);

        netz.setNeuronFkt(0, 0, 0);
        netz.setNeuronFkt(0, 1, 0);
        netz.setNeuronFkt(1, 0, 0);
        netz.setNeuronFkt(1, 1, 0);

        //Neurons Layer 0
        netz.setNeuronWeights(0, 0, 0, -0.5);
        netz.setNeuronWeights(0, 0, 1, -0.4);
        netz.setBiasWeights(0, 0, -0.3);

        netz.setNeuronWeights(0, 1, 0, -0.2);
        netz.setNeuronWeights(0, 1, 1, -0.1);
        netz.setBiasWeights(0, 1, 0.1);

        //Neurons Layer 1
        netz.setNeuronWeights(1, 0, 0, 0.1);
        netz.setNeuronWeights(1, 0, 1, 0.2);
        netz.setBiasWeights(1, 0, 0.3);

        netz.setNeuronWeights(1, 1, 0, 0.4);
        netz.setNeuronWeights(1, 1, 1, 0.5);
        netz.setBiasWeights(1, 1, 0.6);


        assertEquals(2.02, Math.round(netz.forwardPass() * 100.0) / 100.0);
    }

    @Test
    public void backwardPassTest() {
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(-10.0);
        eingabe.add(10.0);
        Netz netz = new Netz( 2, 2);
        netz.init(eingabe);
        netz.setBias(1.0);
        netz.setLearningRate(0.00001);

        System.out.println(netz.forwardPass());
        for (int i = 0; i < 5; i++) {
            netz.backwardPass(2.02);
        }
        System.out.println(netz.forwardPass());
    }
}