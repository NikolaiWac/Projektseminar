import java.util.ArrayList;

public class TestDatenSatz {

        public static void main(String[] args) {


            // x = Fahrzeit in Stunden, y = gefahrene Kilometer bei 2 km/h
            double[] xTrain = {0.0, 0.5, 1.0, 1.5, 2.0};
            double[] yTrain = {0.0, 1.0, 2.0, 3.0, 4.0};

            // Eingabeliste für das Netz (wird später pro Sample überschrieben)
            ArrayList<Double> eingabe = new ArrayList<>();
            // Platz für einen Input-Wert (wird gleich mehrfach überschrieben)
            eingabe.add(xTrain[0]);

            // Netz erzeugen:
            Netz netz = new Netz(3, 1);
            netz.init(eingabe);
            netz.setBias(1.0);
            netz.setLearningRate(0.05);  // ggf. anpassen, falls Lernen zu langsam/schnell

            //  Training
            int epochen = 2000;

            for (int epoch = 0; epoch < epochen; epoch++) {
                double mse = 0.0; // Mean Squared Error über alle Trainingsbeispiele

                for (int i = 0; i < xTrain.length; i++) {
                    // Aktuelles Trainingsbeispiel in die Eingabestruktur schreiben
                    eingabe.set(0, xTrain[i]);

                    // Vorwärtsdurchlauf
                    double output = netz.forwardPass();
                    double fehler = yTrain[i] - output;
                    mse += fehler * fehler;

                    // Rückwärtsdurchlauf (Gewichte anpassen)
                    netz.backwardPass(yTrain[i]);
                }

                mse /= xTrain.length;

                if (epoch % 200 == 0) {
                    System.out.println("Epoche " + epoch + "  |  MSE = " + mse);
                }
            }

            // Ergebnisse auf den Trainingsdaten anzeigen
            System.out.println("\nTrainingsdaten vs. Netzvorhersage:");
            for (int i = 0; i < xTrain.length; i++) {
                eingabe.set(0, xTrain[i]);
                double output = netz.forwardPass();
                System.out.printf("x = %.2f  |  y_soll = %.2f  |  y_vorhergesagt = %.4f%n",
                        xTrain[i], yTrain[i], output);
            }

            // Netz auf neuen Werten testen
            double[] xTest = {0.25, 0.75, 1.25, 1.75};
            System.out.println("\nNeue Testwerte:");
            for (double x : xTest) {
                eingabe.set(0, x);
                double yPred = netz.forwardPass();
                System.out.printf("x = %.2f  ->  y ≈ %.4f%n", x, yPred);
            }
        }
    }


