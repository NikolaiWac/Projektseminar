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
}