import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

public class MnistLoader {

    public static class MnistSet {
        public final double[][] images; // [n][784], normalized to [0,1]
        public final int[] labels;      // [n]
        public final int width;
        public final int height;

        public MnistSet(double[][] images, int[] labels, int width, int height) {
            this.images = images;
            this.labels = labels;
            this.width = width;
            this.height = height;
        }
    }

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES  = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS  = "t10k-labels-idx1-ubyte.gz";

    // Primary download host (mirrored and stable)
    private static final String GCS_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/";

    public static void ensureMnistFiles(String dir) throws IOException {
        Path d = Path.of(dir);
        if (!Files.exists(d)) Files.createDirectories(d);
        downloadIfMissing(dir, TRAIN_IMAGES);
        downloadIfMissing(dir, TRAIN_LABELS);
        downloadIfMissing(dir, TEST_IMAGES);
        downloadIfMissing(dir, TEST_LABELS);
    }

    private static void downloadIfMissing(String dir, String file) throws IOException {
        Path target = Path.of(dir, file);
        if (Files.exists(target)) return;
        String url = GCS_BASE + file;
        System.out.println("Downloading MNIST file: " + url);
        try (InputStream in = new URL(url).openStream()) {
            Files.copy(in, target);
        } catch (IOException ex) {
            throw new IOException("Failed to download " + file + ": " + ex.getMessage(), ex);
        }
    }

    public static MnistSet loadTraining(String dir) throws IOException {
        return load(dir, true);
    }

    public static MnistSet loadTest(String dir) throws IOException {
        return load(dir, false);
    }

    private static MnistSet load(String dir, boolean training) throws IOException {
        String imgName = training ? TRAIN_IMAGES : TEST_IMAGES;
        String lblName = training ? TRAIN_LABELS : TEST_LABELS;

        File imgFile = Path.of(dir, imgName).toFile();
        File lblFile = Path.of(dir, lblName).toFile();

        if (!imgFile.exists() || !lblFile.exists()) {
            throw new FileNotFoundException("MNIST files not found in '" + dir + "'. Expected: " + imgName + ", " + lblName);
        }

        try (DataInputStream imgIn = new DataInputStream(buffered(gzipOrFileStream(imgFile)));
             DataInputStream lblIn = new DataInputStream(buffered(gzipOrFileStream(lblFile)))) {

            int imgMagic = imgIn.readInt(); // 0x00000803
            int numImages = imgIn.readInt();
            int rows = imgIn.readInt();
            int cols = imgIn.readInt();

            int lblMagic = lblIn.readInt(); // 0x00000801
            int numLabels = lblIn.readInt();

            if (numImages != numLabels) {
                throw new IOException("Images count " + numImages + " != labels count " + numLabels);
            }

            double[][] images = new double[numImages][rows * cols];
            int[] labels = new int[numImages];

            // Read all labels first
            for (int i = 0; i < numLabels; i++) {
                int lab = lblIn.readUnsignedByte();
                labels[i] = lab;
            }

            // Read image pixels
            byte[] buffer = new byte[rows * cols];
            for (int i = 0; i < numImages; i++) {
                int read = imgIn.readNBytes(buffer, 0, buffer.length);
                if (read != buffer.length) {
                    throw new EOFException("Unexpected EOF in MNIST images at sample " + i);
                }
                double[] flat = images[i];
                for (int p = 0; p < buffer.length; p++) {
                    int unsigned = buffer[p] & 0xFF;
                    flat[p] = unsigned / 255.0; // normalize
                }
            }

            return new MnistSet(images, labels, cols, rows);
        }
    }

    private static InputStream gzipOrFileStream(File f) throws IOException {
        InputStream base = new FileInputStream(f);
        if (f.getName().toLowerCase().endsWith(".gz")) {
            return new GZIPInputStream(base);
        }
        return base;
    }

    private static BufferedInputStream buffered(InputStream in) {
        return new BufferedInputStream(in, 1 << 20);
    }
}
