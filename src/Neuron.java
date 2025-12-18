public class Neuron {
    private int layerPos;
    private double out;
    private double in;
    private double biasWeight;
    private double[] furtherFktInfo;
    int aktFkt;
    private double[] weights;
    int inputNum;

    public int getAktFkt() {
        return aktFkt;
    }

    public void setAktFkt(int aktFkt) {
        this.aktFkt = aktFkt;
    }

    public void setAktFkt(int aktFkt,  double[] furtherInfo) {
        this.aktFkt = aktFkt;
        furtherFktInfo = furtherInfo;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(int input, double weights) {
        this.weights[input] = weights;
    }

    public void setBiasWeight (double biasWeight) {
        this.biasWeight = biasWeight;
    }

    //input number damit weights direkt beim erstellen zugewiesen werden können
    public Neuron(int aktFkt, int inputNum, int layerPos) {
        this.layerPos = layerPos;
        this.aktFkt = aktFkt;
        weights = new double[inputNum];
        this.inputNum = inputNum;
        asignrandomWeights();
    }

    //Das ist wahrscheinlich temp bis wir mit dem eigentlichen
    //lernen anfangen
    public void asignrandomWeights() {
        // He initialization (optimal for ReLU): uniform in [-limit, +limit]
        // with limit = sqrt(2.0 / fan_in)
        double limit = Math.sqrt(2.0 / Math.max(1.0, (double) inputNum));
        java.util.concurrent.ThreadLocalRandom rnd = java.util.concurrent.ThreadLocalRandom.current();
        for (int i = 0; i < inputNum; i++) {
            weights[i] = rnd.nextDouble(-limit, limit);
        }
        // Keep biasWeight at 0.0 (biases are not updated by current training code)
    }

    //Berechnet den Output des Knoten
    public double outputFkt(double[] input, double bias) {
        double[] w = this.weights;
        double s = 0.0;
        int len = input.length; // equals w.length
        for (int j = 0; j < len; j++) {
            s += input[j] * w[j];
        }
        s += bias * biasWeight;
        in = s;
        out = ActFuntions.funkcionSelect(s, aktFkt, furtherFktInfo);
        return out;
    }

    public double getOut() {
        return out;
    }

    public double getIn() {
        return in;
    }

    public int getLayerPos(){
        return layerPos;
    }
}
