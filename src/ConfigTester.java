import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Class that tests different configurations for the neural network
 * and ranks them by accuracy on the MNIST dataset.
 */
public class ConfigTester {
    private static final String MNIST_DIR = "mnist";
    private static final int INPUT_SIZE = 784;
    private static final int NUM_CLASSES = 10;

    // ANSI color codes for console output
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_GREEN = "\u001B[32m";

    /**
     * Represents a single network configuration and its result.
     */
    static class Config {
        int[] hiddenLayers;
        int epochs;
        double learningRate;
        double accuracy;

        public Config(int[] hiddenLayers, int epochs, double learningRate) {
            this.hiddenLayers = hiddenLayers;
            this.epochs = epochs;
            this.learningRate = learningRate;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("LR: ").append(String.format(Locale.ROOT, "%.4f", learningRate));
            sb.append(", Epochs: ").append(epochs);
            sb.append(", Hidden Layers: ").append(Arrays.toString(hiddenLayers));
            sb.append(" | Accuracy: ").append(String.format(Locale.ROOT, "%.2f%%", accuracy * 100));
            return sb.toString();
        }
    }

    public static void main(String[] args) {
        try {
            System.out.println(ANSI_CYAN + "====================================================" + ANSI_RESET);
            System.out.println(ANSI_CYAN + "   Neural Network Configuration Tester (MNIST)" + ANSI_RESET);
            System.out.println(ANSI_CYAN + "====================================================" + ANSI_RESET);
            System.out.println("This program tests random network configurations indefinitely.");
            System.out.println("Press " + ANSI_YELLOW + "ENTER" + ANSI_RESET + " to stop the search and show final results.");
            System.out.println();

            // Ensure data is available
            MnistLoader.ensureMnistFiles(MNIST_DIR);
            System.out.println("Loading MNIST training data...");
            MnistLoader.MnistSet trainSet = MnistLoader.loadTraining(MNIST_DIR);
            System.out.println("Loading MNIST test data...");
            MnistLoader.MnistSet testSet = MnistLoader.loadTest(MNIST_DIR);

            List<Config> results = new ArrayList<>();
            Random rnd = new Random();
            AtomicBoolean running = new AtomicBoolean(true);

            // Thread to listen for user input (Enter key)
            Thread inputThread = new Thread(() -> {
                try {
                    // Wait for any input
                    System.in.read();
                    running.set(false);
                    System.out.println(ANSI_YELLOW + "\n[STOP SIGNAL] Completing current configuration and shutting down..." + ANSI_RESET);
                } catch (IOException e) {
                    // Ignore
                }
            });
            inputThread.setDaemon(true);
            inputThread.start();

            int testCount = 1;
            while (running.get()) {
                Config config = generateRandomConfig(rnd);
                System.out.println(ANSI_CYAN + "\n[Test #" + testCount + "] " + ANSI_RESET + "Testing: " + config);
                
                try {
                    double acc = trainAndEvaluate(config, trainSet, testSet, running);
                    
                    if (acc >= 0) {
                        config.accuracy = acc;
                        results.add(config);
                        
                        // Sort by accuracy descending
                        results.sort((c1, c2) -> Double.compare(c2.accuracy, c1.accuracy));
                        
                        System.out.println("\n" + ANSI_GREEN + "Configuration finished. Current Top 5:" + ANSI_RESET);
                        for (int i = 0; i < Math.min(5, results.size()); i++) {
                            System.out.println("  " + (i + 1) + ". " + results.get(i));
                        }
                    } else {
                        System.out.println("\nTest interrupted by user.");
                    }
                } catch (Exception e) {
                    System.err.println("Error during training: " + e.getMessage());
                }
                testCount++;
            }

            // Print final summary
            System.out.println("\n" + ANSI_CYAN + "====================================================" + ANSI_RESET);
            System.out.println(ANSI_CYAN + "                FINAL RANKING" + ANSI_RESET);
            System.out.println(ANSI_CYAN + "====================================================" + ANSI_RESET);
            
            if (results.isEmpty()) {
                System.out.println("No configurations were completed.");
            } else {
                for (int i = 0; i < results.size(); i++) {
                    String prefix = (i < 3) ? ANSI_GREEN : "";
                    String suffix = (i < 3) ? ANSI_RESET : "";
                    System.out.printf("%s%3d. %s%s\n", prefix, (i + 1), results.get(i), suffix);
                }
            }
            System.out.println(ANSI_CYAN + "====================================================" + ANSI_RESET);

        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Generates a random configuration for the network.
     */
    private static Config generateRandomConfig(Random rnd) {
        // Number of hidden layers: 1 to 5
        //int numHiddenLayers = rnd.nextInt(1) + 1;
        int numHiddenLayers = 1;
        int[] hiddenLayers = new int[numHiddenLayers];
        for (int i = 0; i < numHiddenLayers; i++) {
            // Neurons: 32 to 512
            hiddenLayers[i] = 32 + rnd.nextInt(481);
        }
        
        // Epochs: 1 to 30
        int epochs = 1 + rnd.nextInt(30);
        
        // Learning rate: 0.001 to 0.2 (log-ish distribution)
        double lr = 0.001 * Math.pow(10, rnd.nextDouble() * 2.3); 
        
        return new Config(hiddenLayers, epochs, lr);
    }

    /**
     * Trains the network with the given config and evaluates its accuracy.
     * Returns -1 if interrupted by the user.
     */
    private static double trainAndEvaluate(Config config, MnistLoader.MnistSet trainSet, MnistLoader.MnistSet testSet, AtomicBoolean running) {
        // Build the network architecture: [Hidden1, Hidden2, ..., Output]
        int[] architecture = new int[config.hiddenLayers.length + 1];
        System.arraycopy(config.hiddenLayers, 0, architecture, 0, config.hiddenLayers.length);
        architecture[architecture.length - 1] = NUM_CLASSES;

        double[] inputRef = new double[INPUT_SIZE];
        Netz netz = new Netz(architecture);
        netz.init(inputRef);
        netz.setBias(1.0);
        netz.setLearningRate(config.learningRate);
        netz.setParallelEnabled(true);

        // Configure activation functions
        // All hidden layers: ReLU (3)
        for (int layer = 0; layer < config.hiddenLayers.length; layer++) {
            setLayerActivation(netz, layer, 3);
        }
        // Output layer: Sigmoid (2)
        setLayerActivation(netz, architecture.length - 1, 2);

        int n = trainSet.images.length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random rnd = new Random();

        // Training loop
        System.out.print("Training epochs: ");
        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            if (!running.get()) return -1;

            // Shuffle training indices
            for (int i = n - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }

            for (int ii = 0; ii < n; ii++) {
                // Check for interruption occasionally
                if (ii % 5000 == 0 && !running.get()) return -1;

                int i = idx[ii];
                System.arraycopy(trainSet.images[i], 0, inputRef, 0, INPUT_SIZE);
                double[] target = oneHot(trainSet.labels[i], NUM_CLASSES);
                netz.backwardPassVector(target);
            }
            System.out.print(epoch + " ");
        }
        System.out.println("- Done.");

        // Evaluation
        int correct = 0;
        int tn = testSet.images.length;
        for (int i = 0; i < tn; i++) {
            System.arraycopy(testSet.images[i], 0, inputRef, 0, INPUT_SIZE);
            double[] pred = netz.forwardPassVector();
            if (argmax(pred) == testSet.labels[i]) {
                correct++;
            }
        }

        return (double) correct / tn;
    }

    /**
     * Sets the activation function for all neurons in a specific layer.
     */
    private static void setLayerActivation(Netz netz, int layer, int fkt) {
        for (int pos = 0; ; pos++) {
            try {
                netz.setNeuronFkt(layer, pos, fkt);
            } catch (ArrayIndexOutOfBoundsException ex) {
                break;
            }
        }
    }

    private static int argmax(double[] v) {
        int bestIdx = 0;
        double best = v[0];
        for (int i = 1; i < v.length; i++) {
            if (v[i] > best) {
                best = v[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static double[] oneHot(int label, int classes) {
        double[] t = new double[classes];
        if (label >= 0 && label < classes) {
            t[label] = 1.0;
        }
        return t;
    }
}
