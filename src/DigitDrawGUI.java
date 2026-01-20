import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class DigitDrawGUI extends JFrame {

    private final Netz netz;
    private final int width;
    private final int height;
    private final double[] inputRef;

    private final DrawPanel drawPanel;
    private final JLabel resultLabel;
    private final JTextArea outputsArea;
    private final JLabel previewLabel;

    public DigitDrawGUI(Netz netz, int width, int height, double[] inputRef) {
        super("Digit Recognizer (MNIST) - Draw with Mouse");
        this.netz = netz;
        this.width = width;
        this.height = height;
        this.inputRef = inputRef;

        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(12, 12));

        drawPanel = new DrawPanel(320, 320);

        // Right side UI
        JPanel right = new JPanel();
        right.setLayout(new BoxLayout(right, BoxLayout.Y_AXIS));

        resultLabel = new JLabel("Prediction: -");
        resultLabel.setFont(resultLabel.getFont().deriveFont(Font.BOLD, 22f));
        resultLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        previewLabel = new JLabel();
        previewLabel.setPreferredSize(new Dimension(140, 140));
        previewLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        previewLabel.setBorder(BorderFactory.createTitledBorder("28x28 Preview"));

        outputsArea = new JTextArea(12, 18);
        outputsArea.setEditable(false);
        outputsArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane scroll = new JScrollPane(outputsArea);
        scroll.setAlignmentX(Component.LEFT_ALIGNMENT);
        scroll.setBorder(BorderFactory.createTitledBorder("Outputs (0..9)"));

        JPanel btnRow = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton predictBtn = new JButton("Predict");
        JButton clearBtn = new JButton("Clear");
        JButton thickerBtn = new JButton("Brush +");
        JButton thinnerBtn = new JButton("Brush -");

        predictBtn.addActionListener(e -> doPredict());
        clearBtn.addActionListener(e -> {
            drawPanel.clear();
            resultLabel.setText("Prediction: -");
            outputsArea.setText("");
            previewLabel.setIcon(null);
        });
        thickerBtn.addActionListener(e -> drawPanel.setBrushSize(drawPanel.getBrushSize() + 2));
        thinnerBtn.addActionListener(e -> drawPanel.setBrushSize(Math.max(2, drawPanel.getBrushSize() - 2)));

        btnRow.add(predictBtn);
        btnRow.add(clearBtn);
        btnRow.add(thickerBtn);
        btnRow.add(thinnerBtn);
        btnRow.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel hint = new JLabel("<html><body style='width: 240px'>Draw a digit (0-9) with the mouse.<br>" +
                "This canvas is <b>white ink on black</b> to match MNIST.<br>" +
                "Tip: draw big, centered.</body></html>");
        hint.setAlignmentX(Component.LEFT_ALIGNMENT);

        right.add(resultLabel);
        right.add(Box.createVerticalStrut(8));
        right.add(hint);
        right.add(Box.createVerticalStrut(10));
        right.add(btnRow);
        right.add(Box.createVerticalStrut(10));
        right.add(previewLabel);
        right.add(Box.createVerticalStrut(10));
        right.add(scroll);

        add(drawPanel, BorderLayout.CENTER);
        add(right, BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void doPredict() {
        BufferedImage src = drawPanel.getImage();
        BufferedImage mnist28 = preprocessToMnist28(src);


        Image previewScaled = mnist28.getScaledInstance(128, 128, Image.SCALE_FAST);
        previewLabel.setIcon(new ImageIcon(previewScaled));

        double[] flat = flattenNormalized(mnist28); // 0..1
        if (flat.length != width * height) {
            JOptionPane.showMessageDialog(this, "Expected " + (width * height) + " inputs, got " + flat.length);
            return;
        }

        System.arraycopy(flat, 0, inputRef, 0, flat.length);
        double[] out = netz.forwardPassVector();
        int pred = argmax(out);

        resultLabel.setText("Prediction: " + pred);
        outputsArea.setText(formatOutputs(out));
    }



    public static BufferedImage preprocessToMnist28(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();

        // Convert to grayscale float
        float[][] g = new float[h][w];
        float maxVal = 0f;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gg = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                float gray = (float) ((0.299 * r) + (0.587 * gg) + (0.114 * b)) / 255f;
                g[y][x] = gray;
                if (gray > maxVal) maxVal = gray;
            }
        }


        if (maxVal < 0.05f) {
            return blackImage(28, 28);
        }


        float thr = 0.10f;
        int minX = w, minY = h, maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (g[y][x] >= thr) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < 0) {
            return blackImage(28, 28);
        }


        int pad = 12;
        minX = Math.max(0, minX - pad);
        minY = Math.max(0, minY - pad);
        maxX = Math.min(w - 1, maxX + pad);
        maxY = Math.min(h - 1, maxY + pad);

        int cropW = Math.max(1, maxX - minX + 1);
        int cropH = Math.max(1, maxY - minY + 1);

        BufferedImage cropped = new BufferedImage(cropW, cropH, BufferedImage.TYPE_INT_RGB);
        Graphics2D gc = cropped.createGraphics();
        gc.drawImage(src, 0, 0, cropW, cropH, minX, minY, maxX + 1, maxY + 1, null);
        gc.dispose();


        int target = 20;
        double scale = Math.min((double) target / cropW, (double) target / cropH);
        int newW = Math.max(1, (int) Math.round(cropW * scale));
        int newH = Math.max(1, (int) Math.round(cropH * scale));

        BufferedImage scaled = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D gs = scaled.createGraphics();
        gs.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        gs.drawImage(cropped, 0, 0, newW, newH, null);
        gs.dispose();


        BufferedImage out = blackImage(28, 28);
        Graphics2D go = out.createGraphics();
        int offX = (28 - newW) / 2;
        int offY = (28 - newH) / 2;
        go.drawImage(scaled, offX, offY, null);
        go.dispose();

        return out;
    }

    private static BufferedImage blackImage(int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    public static double[] flattenNormalized(BufferedImage image28) {
        int w = image28.getWidth();
        int h = image28.getHeight();
        double[] flat = new double[w * h];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image28.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double gray = (0.299 * r) + (0.587 * g) + (0.114 * b);
                flat[idx++] = gray / 255.0;
            }
        }
        return flat;
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

    private static String formatOutputs(double[] out) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < out.length; i++) {
            sb.append(i).append(": ").append(String.format(java.util.Locale.ROOT, "%.6f", out[i])).append("\n");
        }


        Integer[] idx = new Integer[out.length];
        for (int i = 0; i < out.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(out[b], out[a]));
        sb.append("\nTop-3:\n");
        for (int k = 0; k < Math.min(3, idx.length); k++) {
            int d = idx[k];
            sb.append("#").append(k + 1).append(" -> ").append(d)
                    .append(" (").append(String.format(java.util.Locale.ROOT, "%.6f", out[d])).append(")\n");
        }
        return sb.toString();
    }



    private static class DrawPanel extends JPanel {
        private final BufferedImage img;
        private int brushSize = 18;
        private int lastX = -1, lastY = -1;

        DrawPanel(int w, int h) {
            setPreferredSize(new Dimension(w, h));
            img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            clear();

            MouseAdapter ma = new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    lastX = e.getX();
                    lastY = e.getY();
                    drawDot(lastX, lastY);
                }

                @Override
                public void mouseDragged(MouseEvent e) {
                    int x = e.getX();
                    int y = e.getY();
                    drawLine(lastX, lastY, x, y);
                    lastX = x;
                    lastY = y;
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    lastX = -1;
                    lastY = -1;
                }
            };
            addMouseListener(ma);
            addMouseMotionListener(ma);
        }

        int getBrushSize() { return brushSize; }
        void setBrushSize(int s) { brushSize = s; }

        BufferedImage getImage() { return img; }

        void clear() {
            Graphics2D g = img.createGraphics();
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, img.getWidth(), img.getHeight());
            g.dispose();
            repaint();
        }

        private void drawDot(int x, int y) {
            Graphics2D g = img.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(Color.WHITE);
            int r = brushSize / 2;
            g.fillOval(x - r, y - r, brushSize, brushSize);
            g.dispose();
            repaint();
        }

        private void drawLine(int x1, int y1, int x2, int y2) {
            Graphics2D g = img.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(Color.WHITE);
            g.setStroke(new BasicStroke(brushSize, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            g.drawLine(x1, y1, x2, y2);
            g.dispose();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(img, 0, 0, null);
        }
    }
}

