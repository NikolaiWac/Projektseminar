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
        for (int i = 0; i < inputNum; i++) {
            weights[i] = Math.random();
        }
    }

    //Berechnet den Output des Knoten
    public double outputFkt(double[] input, double bias) {
        double sum = 0;
        //es wird in der Formel Bias gebraucht, keine Ahnung ob
        //wir den schon jetz brauchen
        for (int j = 0; j < input.length; j++) {
            sum += input[j] * weights[j];
        }
        sum += bias * biasWeight;
        in = sum;
        out = ActFuntions.funkcionSelect(sum, aktFkt, furtherFktInfo);
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
