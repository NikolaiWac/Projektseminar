import java.util.ArrayList;

public class Neuron {
    private int layerPos;
    private double out;
    private double in;
    private double biasWeight;
    private ArrayList<Double> furtherFktInfo;
    int aktFkt;
    private ArrayList<Double> weights;
    int inputNum;

    public int getAktFkt() {
        return aktFkt;
    }

    public void setAktFkt(int aktFkt) {
        this.aktFkt = aktFkt;
    }

    public void setAktFkt(int aktFkt,  ArrayList<Double> furtherInfo) {
        this.aktFkt = aktFkt;
        furtherFktInfo = furtherInfo;
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public void setWeights(int input, double weights) {
        this.weights.set(input, weights);
    }

    public void setBiasWeight (double biasWeight) {
        this.biasWeight = biasWeight;
    }

    //input number damit weights direkt beim erstellen zugewiesen werden können
    public Neuron(int aktFkt, int inputNum, int layerPos) {
        this.layerPos = layerPos;
        this.aktFkt = aktFkt;
        weights = new ArrayList<>();
        this.inputNum = inputNum;
        asignrandomWeights();
    }

    //Das ist wahrscheinlich temp bis wir mit dem eigentlichen
    //lernen anfangen
    public void asignrandomWeights() {
        for (int i = 0; i < inputNum; i++) {
            weights.add(Math.random());
        }
    }

    //Berechnet den Output des Knoten
    public double outputFkt(ArrayList<Double> input, double bias) {
        double sum = 0;
        //es wird in der Formel Bias gebraucht, keine Ahnung ob
        //wir den schon jetz brauchen
        for (int j = 0; j < input.size(); j++) {
            sum += input.get(j) * weights.get(j);
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
