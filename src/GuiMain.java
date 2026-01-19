import java.util.Locale;
import java.util.Random;

public class GuiMain {

    public static void main(String[] args) {
        try {
            // 1) MNIST sicherstellen & laden
            String mnistDir = "mnist";
            System.out.println("Preparing MNIST dataset (will download if not found)...");
            MnistLoader.ensureMnistFiles(mnistDir);
            MnistLoader.MnistSet trainSet = MnistLoader.loadTraining(mnistDir);
            MnistLoader.MnistSet testSet = MnistLoader.loadTest(mnistDir);

            int width = trainSet.width;   // 28
            int height = trainSet.height; // 28
            int inputSize = width * height; // 784
            int numClasses = 10;

            // 2) Netzwerk bauen
            double[] inputRef = new double[inputSize]; // shared input buffer
            Netz netz = new Netz(300, 10);
            netz.init(inputRef);
            netz.setBias(1.0);
            netz.setLearningRate(0.02);
            netz.setParallelEnabled(true);

            // Hidden layer: ReLU (ID 3)
            for (int pos = 0; ; pos++) {
                try {
                    netz.setNeuronFkt(0, pos, 3);
                } catch (ArrayIndexOutOfBoundsException ex) {
                    break;
                }
            }
            // Output layer: Sigmoid (ID 2)
            for (int pos = 0; ; pos++) {
                try {
                    netz.setNeuronFkt(1, pos, 2);
                } catch (ArrayIndexOutOfBoundsException ex) {
                    break;
                }
            }

            // 3) Training
            int n = trainSet.images.length;
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Random rnd = new Random(1234);

            int epochs = 10; // für bessere Erkennung erhöhen
            for (int epoch = 1; epoch <= epochs; epoch++) {
                // shuffle
                for (int i = n - 1; i > 0; i--) {
                    int j = rnd.nextInt(i + 1);
                    int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                }

                double mse = 0.0;
                int correct = 0;

                for (int ii = 0; ii < n; ii++) {
                    int i = idx[ii];
                    double[] x = trainSet.images[i];
                    int label = trainSet.labels[i];
                    double[] y = oneHot(label, numClasses);

                    System.arraycopy(x, 0, inputRef, 0, inputSize);

                    double[] predVec = netz.forwardPassVector();
                    mse += squaredError(predVec, y);
                    if (argmax(predVec) == label) correct++;

                    netz.backwardPassVector(y);
                }

                mse /= n;
                double acc = (double) correct / n;
                System.out.println("Epoch " + epoch + " / " + epochs
                        + " | mse = " + String.format(Locale.ROOT, "%.6f", mse)
                        + " | train acc = " + String.format(Locale.ROOT, "%.2f%%", acc * 100));
            }

            // 4) Test-Accuracy kurz ausgeben
            {
                int tn = testSet.images.length;
                int testCorrect = 0;
                for (int i = 0; i < tn; i++) {
                    double[] x = testSet.images[i];
                    int label = testSet.labels[i];
                    System.arraycopy(x, 0, inputRef, 0, inputSize);
                    int pred = argmax(netz.forwardPassVector());
                    if (pred == label) testCorrect++;
                }
                double testAcc = (double) testCorrect / tn;
                System.out.println("Test accuracy: " + String.format(Locale.ROOT, "%.2f%%", testAcc * 100));
            }

            // 5) GUI starten
            javax.swing.SwingUtilities.invokeLater(() -> new DigitDrawGUI(netz, width, height, inputRef));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double[] oneHot(int label, int numClasses) {
        double[] v = new double[numClasses];
        v[label] = 1.0;
        return v;
    }

    private static double squaredError(double[] pred, double[] target) {
        double s = 0.0;
        for (int i = 0; i < pred.length; i++) {
            double d = target[i] - pred[i];
            s += d * d;
        }
        return s / pred.length;
    }

    private static int argmax(double[] v) {
        int best = 0;
        double bestVal = v[0];
        for (int i = 1; i < v.length; i++) {
            if (v[i] > bestVal) {
                bestVal = v[i];
                best = i;
            }
        }
        return best;
    }
}
