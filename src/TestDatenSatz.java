public class TestDatenSatz {

        public static void main(String[] args) {


            double[] qmReell    = {50.0, 70.0, 90.0};
            int[]    zimmerReell= {2, 3, 4};
            double[] mieteReell = {550.0, 750.0, 950.0};

            int n = qmReell.length;


            double[][] xTrain = new double[n][2];
            double[]   yTrain = new double[n];

            for (int i = 0; i < n; i++) {
                xTrain[i][0] = qmReell[i] / 100.0;
                xTrain[i][1] = zimmerReell[i] / 10.0;
                yTrain[i]    = mieteReell[i] / 1000.0;
            }

            double[] eingabe = new double[2];
            eingabe[0] = xTrain[0][0];
            eingabe[1] = xTrain[0][1];

            // Netz erzeugen
            Netz netz = new Netz(2, 6, 1);
            netz.init(eingabe);

            netz.setBias(1.0);
            netz.setLearningRate(0.01);


            int epochen = 5000;

            for (int epoch = 0; epoch < epochen; epoch++) {
                double mse = 0.0;

                for (int i = 0; i < n; i++) {
                    // Aktuelle Wohnung
                    eingabe[0] = xTrain[i][0];
                    eingabe[1] = xTrain[i][1];

                    // Vorwärtsdurchlauf
                    double output = netz.forwardPass();
                    double fehler = yTrain[i] - output;
                    mse += fehler * fehler;

                    // Rückwärtsdurchlauf
                    netz.backwardPass(yTrain[i]);
                }

                mse /= n;

                if (epoch % 500 == 0) {
                    System.out.println("Epoche " + epoch + "  |  MSE = " + mse);
                }
            }


            System.out.println("\nTrainingsdaten vs. Netzvorhersage (denormalisiert):");
            for (int i = 0; i < n; i++) {
                eingabe[0] = xTrain[i][0];
                eingabe[1] = xTrain[i][1];

                double outputNorm = netz.forwardPass();
                double outputEuro = outputNorm * 1000.0;

                System.out.printf(
                        "Wohnung %c: %.0f m², %d Zimmer  |  Miete_soll = %.0f €  |  Miete_Netz ≈ %.2f €%n",
                        ('A' + i),
                        qmReell[i],
                        zimmerReell[i],
                        mieteReell[i],
                        outputEuro
                );
            }


            System.out.println("\nNeue Wohnungen (Netzvorhersage):");

            // Beispiel: 60 m², 2 Zimmer
            testWohnung(netz, eingabe, 60.0, 2);

            // Beispiel: 80 m², 3 Zimmer
            testWohnung(netz, eingabe, 80.0, 3);

            // Beispiel: 100 m², 4 Zimmer
            testWohnung(netz, eingabe, 100.0, 4);
        }


        private static void testWohnung(Netz netz, double[] eingabe,
                                        double qm, int zimmer) {
            double qmNorm = qm / 100.0;
            double zimmerNorm = zimmer / 10.0;

            eingabe[0] = qmNorm;
            eingabe[1] = zimmerNorm;

            double outputNorm = netz.forwardPass();
            double outputEuro = outputNorm * 1000.0;

            System.out.printf(
                    "Testwohnung: %.0f m², %d Zimmer  ->  geschätzte Miete ≈ %.2f €%n",
                    qm, zimmer, outputEuro
            );
        }
    }
