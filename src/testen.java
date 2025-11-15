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
    public void backwardPassTest2() {
        ArrayList<Double> eingabe = new ArrayList<>();
        eingabe.add(0.5);
        eingabe.add(0.3);

        // Erstelle ein Netz mit 2 Inputs, 1 Hidden Neuron und 1 Output Neuron
        Netz netz = new Netz(1, 1);
        netz.init(eingabe);
        netz.setBias(1.0);
        netz.setLearningRate(0.1); // Lernrate

        // Setze alle Aktivierungsfunktionen auf Sigmoid (ID 2), da diese differenzierbar ist
        netz.setNeuronFkt(0, 0, 2); // Hidden Layer
        netz.setNeuronFkt(1, 0, 2); // Output Layer

        // Setze feste Startgewichte für deterministisches Testen
        // Hidden Neuron (Layer 0, Neuron 0)
        netz.setNeuronWeights(0, 0, 0, 0.2); // Gewicht von Input 0
        netz.setNeuronWeights(0, 0, 1, 0.4); // Gewicht von Input 1
        netz.setBiasWeights(0, 0, 0.1);     // Bias Gewicht

        // Output Neuron (Layer 1, Neuron 0)
        netz.setNeuronWeights(1, 0, 0, 0.5); // Gewicht von Hidden Neuron 0
        netz.setBiasWeights(1, 0, 0.3);     // Bias Gewicht

        // Definiere ein Ziel
        double target = 0.85;

        // 1. Berechne den Output VOR dem Training
        double outputBefore = netz.forwardPass();
        double errorBefore = Math.abs(target - outputBefore);
        System.out.println("Output vor Training: " + outputBefore);
        System.out.println("Fehler vor Training: " + errorBefore);

        // 2. Trainiere das Netz für 100 Iterationen
        for (int i = 0; i < 100; i++) {
            netz.backwardPass(target);
        }

        // 3. Berechne den Output NACH dem Training
        double outputAfter = netz.forwardPass();
        double errorAfter = Math.abs(target - outputAfter);
        System.out.println("Output nach Training: " + outputAfter);
        System.out.println("Fehler nach Training: " + errorAfter);

        // 4. Prüfe, ob der Fehler kleiner geworden ist
        assertTrue(errorAfter < errorBefore, "Der Fehler sollte nach dem Training kleiner sein.");
    }
}