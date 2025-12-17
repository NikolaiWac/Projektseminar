import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class Main {

    // ANSI color codes for console output
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";

    public static void main(String[] args) {
        try {
            // 1) Ensure MNIST data exists and load it
            String mnistDir = "mnist";
            System.out.println("Preparing MNIST dataset (will download if not found)...");
            MnistLoader.ensureMnistFiles(mnistDir);
            MnistLoader.MnistSet trainSet = MnistLoader.loadTraining(mnistDir);
            // Load MNIST test set for evaluation after training
            MnistLoader.MnistSet testSet = MnistLoader.loadTest(mnistDir);

            int width = trainSet.width;   // 28
            int height = trainSet.height; // 28
            int inputSize = width * height; // 784
            int numClasses = 10;

            // 2) Build network: input (784) -> hidden(300) -> output(10)
            double[] inputRef = new double[inputSize]; // mutable buffer referenced by the net
            Netz netz = new Netz(300, 10);
            netz.init(inputRef);
            netz.setBias(1.0);
            netz.setLearningRate(0.02);
            // Enable multi-core parallel computation
            netz.setParallelEnabled(true);

            // Set activations per layer:
            // Hidden layer (index 0): ReLU (ID 3)
            for (int pos = 0; ; pos++) {
                try {
                    netz.setNeuronFkt(0, pos, 3);
                } catch (ArrayIndexOutOfBoundsException ex) {
                    break;
                }
            }
            // Output layer (index 1): Sigmoid (ID 2)
            for (int pos = 0; ; pos++) {
                try {
                    netz.setNeuronFkt(1, pos, 2);
                } catch (ArrayIndexOutOfBoundsException ex) {
                    break;
                }
            }

            // 3) Prepare shuffled indices to iterate training samples
            int n = trainSet.images.length;
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Random rnd = new Random(1234);

            // 4) Train for a few epochs (adjust as needed)
            int epochs = 10; // keep small for a quick start; increase for better accuracy
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

                    // copy sample into the shared input buffer
                    System.arraycopy(x, 0, inputRef, 0, inputSize);

                    double[] predVec = netz.forwardPassVector();
                    mse += squaredError(predVec, y);
                    if (argmax(predVec) == label) correct++;
                    netz.backwardPassVector(y);
                }

                mse /= n;
                double acc = (double) correct / n;
                System.out.println("Epoch " + epoch + " / " + epochs + " | MSE = " + mse + " | Train acc = " + String.format(Locale.ROOT, "%.2f%%", acc * 100));
            }

            // 5) Evaluate on the official MNIST test set
            {
                final boolean PRINT_TEST_DETAILS = true; // prints: desired label, full outputs, and predicted label
                int tn = testSet.images.length;
                double testMse = 0.0;
                int testCorrect = 0;
                for (int i = 0; i < tn; i++) {
                    double[] x = testSet.images[i];
                    int label = testSet.labels[i];
                    double[] y = oneHot(label, numClasses);

                    // copy sample into the shared input buffer
                    System.arraycopy(x, 0, inputRef, 0, inputSize);

                    double[] predVec = netz.forwardPassVector();
                    int pred = argmax(predVec);
                    testMse += squaredError(predVec, y);
                    if (pred == label) testCorrect++;
                    if (PRINT_TEST_DETAILS) {
                        String line =
                                "Test sample " + i +
                                " | desired=" + label +
                                " | predicted=" + pred +
                                " | outputs=" + vectorToString(predVec);
                        if (pred == label) {
                            System.out.println(ANSI_GREEN + line + ANSI_RESET);
                        } else {
                            System.out.println(ANSI_RED + line + ANSI_RESET);
                        }
                    }
                }
                testMse /= tn;
                double testAcc = (double) testCorrect / tn;
                System.out.println("\nMNIST test set: MSE = " + testMse + " | Accuracy = " + String.format(Locale.ROOT, "%.2f%%", testAcc * 100));
            }

            // 6) Optional: predict user's PNGs from the 'data' folder (if present).
            // The app no longer requires a 'data' folder. If it's missing or empty, we simply skip this part.
            File dataDir = new File("data");
            if (dataDir.exists() && dataDir.isDirectory()) {
                List<File> testPngs = new ArrayList<>();
                try {
                    Files.list(dataDir.toPath())
                            .map(Path::toFile)
                            .filter(File::isFile)
                            .filter(f -> hasImageExtension(f.getName()))
                            .sorted(Comparator.comparing(File::getName))
                            .forEach(testPngs::add);
                } catch (IOException e) {
                    System.out.println("Skipping custom PNG predictions (failed to list 'data'): " + e.getMessage());
                    testPngs = null;
                }

                if (testPngs != null && !testPngs.isEmpty()) {
                    System.out.println("\nPredictions for images in data/ (resized to 28x28 if necessary):\n");
                    for (File f : testPngs) {
                        try {
                            double[] flat = PNGArr.loadAndFlattenNormalized(f.getAbsolutePath(), width, height);
                            System.arraycopy(flat, 0, inputRef, 0, inputSize);
                            double[] out = netz.forwardPassVector();
                            int pred = argmax(out);
                            System.out.println(f.getName() + " -> predicted digit: " + pred + " | outputs: " + vectorToString(out));
                        } catch (IOException e) {
                            System.out.println("Skipping '" + f.getName() + "': " + e.getMessage());
                        }
                    }
                } else if (testPngs != null) {
                    System.out.println("No PNG images found in 'data'. Skipping custom predictions.");
                }
            } else {
                // Silently skip to avoid requiring a 'data' folder
                // Uncomment the next line if you prefer a notice:
                // System.out.println("No 'data' folder detected. Skipping custom PNG predictions.");
            }

        } catch (Exception e) {
            System.out.println("Unexpected error: " + e.getMessage());
        }
    }

    private static double squaredError(double[] a, double[] b) {
        double s = 0.0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s / n;
    }

    private static int argmax(double[] v) {
        int bestIdx = 0;
        double best = v[0];
        for (int i = 1; i < v.length; i++) {
            if (v[i] > best) { best = v[i]; bestIdx = i; }
        }
        return bestIdx;
    }

    private static double[] oneHot(int label, int classes) {
        double[] t = new double[classes];
        if (label >= 0 && label < classes) t[label] = 1.0;
        return t;
    }

    private static boolean hasImageExtension(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".png");
    }

    private static String vectorToString(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format(Locale.ROOT, "%.3f", v[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}
