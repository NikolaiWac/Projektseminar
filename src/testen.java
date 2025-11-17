import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

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

        Netz netz = new Netz(2, 6, 1);
        netz.init(eingabe);
        netz.setBias(1.0);
        netz.setLearningRate(0.001);

        double target = 4.02;

        double before = netz.forwardPass();
        // Datei für Excel (CSV) vorbereiten und nach jedem Backward-Pass den Forward-Wert protokollieren
        Path csvPath = Paths.get(System.getProperty("user.dir"), "backward_pass_forward_values.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(
                csvPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        )) {
            // Header schreiben
            writer.write("Iteration;ForwardPass\n");

            for (int i = 1; i <= 1000; i++) {
                netz.backwardPass(target);
                double current = netz.forwardPass();
                // Semikolon als Trennzeichen verwenden (Excel-kompatibel in vielen Ländereinstellungen)
                writer.write(i + ";" + current + "\n");
            }
            writer.flush();
        } catch (IOException e) {
            fail("Konnte CSV-Datei nicht schreiben: " + e.getMessage());
        }
        double after = netz.forwardPass();

        // Test: Output nach Training soll näher am Zielwert sein
        double diffBefore = Math.abs(before - target);
        double diffAfter = Math.abs(after - target);

        System.out.println("Vorher: " + before + "  Nachher: " + after);
        assertTrue(diffAfter < diffBefore);
    }
}