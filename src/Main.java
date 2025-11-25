import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public class Main {

    private static class Sample {
        final double[] x;
        final double y; // 0.0 or 1.0
        Sample(double[] x, double y) { this.x = x; this.y = y; }
    }

    public static void main(String[] args) {
        try {
            // 1) Load training data from project "data" folder
            File dataDir = new File("data");
            if (!dataDir.exists() || !dataDir.isDirectory()) {
                System.out.println("No 'data' folder found in the project root. Please create it and add labeled PNG images named like '123 0.png' or '123 1.png'.");
                return;
            }

            List<File> imageFiles = new ArrayList<>();
            try {
                Files.list(dataDir.toPath())
                        .map(Path::toFile)
                        .filter(File::isFile)
                        .filter(f -> hasImageExtension(f.getName()))
                        .sorted(Comparator.comparing(File::getName))
                        .forEach(imageFiles::add);
            } catch (IOException e) {
                System.out.println("Failed to list files in 'data' folder: " + e.getMessage());
                return;
            }

            if (imageFiles.isEmpty()) {
                System.out.println("The 'data' folder is empty. Add PNG images named like '12 0.png' or '45 1.png'. For example: data\\1 0.png, data\\2 1.png");
                return;
            }

            // Peek the first image to determine input size
            String firstPath = imageFiles.get(0).getAbsolutePath();
            double[][] firstImg = PNGArr.PNGtoArray(firstPath);
            int height = firstImg.length;
            int width = firstImg[0].length;
            int inputSize = width * height;

            // 2) Prepare dataset (flatten + normalize to [0,1])
            List<Sample> dataset = new ArrayList<>();
            for (File f : imageFiles) {
                Double label = parseLabelFromFilename(f.getName());
                if (label == null) {
                    // Skip files that don't follow the naming rule
                    continue;
                }
                try {
                    double[][] img = PNGArr.PNGtoArray(f.getAbsolutePath());
                    if (img.length != height || img[0].length != width) {
                        System.out.println("Skipping '" + f.getName() + "' due to mismatching image size.");
                        continue;
                    }
                    double[] flat = flattenAndNormalize(img);
                    dataset.add(new Sample(flat, label));
                } catch (IOException e) {
                    System.out.println("Failed to read '" + f.getName() + "': " + e.getMessage());
                }
            }

            if (dataset.isEmpty()) {
                System.out.println("No valid training images found. Ensure filenames look like '123 0.png' or '123 1.png'.");
                return;
            }

            // 3) Build network: input -> hidden(32) -> output(1)
            double[] inputRef = new double[inputSize]; // mutable buffer referenced by the net
            Netz netz = new Netz(32, 1);
            netz.init(inputRef);
            netz.setBias(1.0);
            netz.setLearningRate(0.01);

            // Set activation to sigmoid (id=2 or any non 0/1) for all neurons to keep outputs in [0,1]
            setAllNeuronsActivationSigmoid(netz);

            // 4) Train
            int epochs = 10; // adjust as needed
            for (int epoch = 1; epoch <= epochs; epoch++) {
                double mse = 0.0;
                for (Sample s : dataset) {
                    // copy sample into the shared input buffer
                    System.arraycopy(s.x, 0, inputRef, 0, inputSize);
                    double pred = netz.forwardPass();
                    double err = s.y - pred;
                    mse += err * err;
                    netz.backwardPass(s.y);
                }
                mse /= dataset.size();
                System.out.println("Epoch " + epoch + " / " + epochs + " | MSE = " + mse);
            }

            // 5) Demo: ask user for an image path (PNG with same size) and predict
            try (Scanner scanner = new Scanner(System.in)) {
                System.out.print("Enter a path to a PNG image to test (or press Enter to skip): ");
                String demoPath = scanner.nextLine();
                if (demoPath != null && !demoPath.trim().isEmpty()) {
                    File demoFile = new File(demoPath.trim());
                    if (!demoFile.exists()) {
                        System.out.println("File not found: " + demoFile.getAbsolutePath());
                        return;
                    }
                    if (!demoFile.getName().toLowerCase(Locale.ROOT).endsWith(".png")) {
                        System.out.println("Please provide a PNG image.");
                        return;
                    }
                    double[][] demoArray = PNGArr.PNGtoArray(demoFile.getAbsolutePath());

                    if (demoArray.length != height || demoArray[0].length != width) {
                        System.out.println("Demo image size doesn't match training images (expected " + width + "x" + height + ").");
                        return;
                    }
                    double[] demoInput = flattenAndNormalize(demoArray);
                    System.arraycopy(demoInput, 0, inputRef, 0, inputSize);
                    double prediction = netz.forwardPass();
                    // Ensure numerical range [0,1]
                    if (prediction < 0) prediction = 0;
                    if (prediction > 1) prediction = 1;
                    System.out.println("Network output (0..1): " + prediction);
                }
            }

        } catch (Exception e) {
            System.out.println("Unexpected error: " + e.getMessage());
        }
    }

    private static boolean hasImageExtension(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        return lower.endsWith(".png");
    }

    private static Double parseLabelFromFilename(String name) {
        // Expected: "<some number> <label>.png" where <label> is 0 or 1
        // Examples: "1 0.png", "25 1.png"
        int dot = name.lastIndexOf('.')
                ;
        String base = dot >= 0 ? name.substring(0, dot) : name;
        String[] parts = base.trim().split(" ");
        if (parts.length < 2) return null;
        String labelStr = parts[parts.length - 1].trim();
        if (labelStr.equals("0")) return 0.0;
        if (labelStr.equals("1")) return 1.0;
        return null;
    }

    private static double[] flattenAndNormalize(double[][] img) {
        int h = img.length;
        int w = img[0].length;
        double[] flat = new double[h * w];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                flat[idx++] = img[y][x] / 255.0; // normalize to [0,1]
            }
        }
        return flat;
    }

    private static void setAllNeuronsActivationSigmoid(Netz netz) {
        // Probe layers and neuron positions with safe bounds.
        // Stop when a layer has no neurons (pos 0 fails).
        for (int layer = 0; layer < 64; layer++) { // generous upper bound
            boolean anyInLayer = false;
            for (int pos = 0; pos < 1024; pos++) { // generous upper bound
                try {
                    netz.setNeuronFkt(layer, pos, 2); // any value other than 0/1 selects sigmoid
                    anyInLayer = true;
                } catch (ArrayIndexOutOfBoundsException ex) {
                    // No more neurons in this layer
                    break;
                }
            }
            if (!anyInLayer) {
                // No such layer
                break;
            }
        }
    }
}
