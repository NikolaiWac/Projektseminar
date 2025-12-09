import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class PNGArr {
    /**
     * Converts a PNG image at the given path to a 2D array of grayscale values.
     * * @param imagePath The file path to the PNG image.
     *
     * @return A 2D double array [height][width] containing grayscale values (0.0 to 255.0).
     * @throws IOException If the file cannot be read.
     */
    public static double[][] PNGtoArray(String imagePath) throws IOException {
        File file = new File(imagePath);
        BufferedImage image = ImageIO.read(file);

        if (image == null) {
            throw new IOException("Unable to decode image file: " + imagePath);
        }

        int width = image.getWidth();
        int height = image.getHeight();

        // Standard convention is [row][col], which maps to [y][x]
        double[][] grayScale = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Get the integer RGB value
                int p = image.getRGB(x, y);

                // Extract Red, Green, and Blue using bitwise shifts
                // (alpha is (p >> 24) & 0xff if needed)
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                // Calculate grayscale using the Luminosity formula (Rec. 601)
                // This matches human perception better than a simple average
                double gray = (0.299 * r) + (0.587 * g) + (0.114 * b);

                // Store the value.
                // To normalize to 0.0 - 1.0, divide 'gray' by 255.0 here.
                grayScale[y][x] = gray;
            }
        }

        return grayScale;
    }

    /**
     * Loads a PNG and returns a flattened, normalized double array of size targetW*targetH.
     * If the image size differs, it will be resized using nearest-neighbor.
     */
    public static double[] loadAndFlattenNormalized(String imagePath, int targetW, int targetH) throws IOException {
        File file = new File(imagePath);
        BufferedImage image = ImageIO.read(file);
        if (image == null) {
            throw new IOException("Unable to decode image file: " + imagePath);
        }

        // Resize if needed
        if (image.getWidth() != targetW || image.getHeight() != targetH) {
            image = resize(image, targetW, targetH);
        }

        double[] flat = new double[targetW * targetH];
        int idx = 0;
        for (int y = 0; y < targetH; y++) {
            for (int x = 0; x < targetW; x++) {
                int p = image.getRGB(x, y);
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                double gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
                flat[idx++] = gray / 255.0; // normalize
            }
        }
        return flat;
    }

    private static BufferedImage resize(BufferedImage src, int w, int h) {
        Image tmp = src.getScaledInstance(w, h, Image.SCALE_REPLICATE);
        BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }
}