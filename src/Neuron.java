import java.util.ArrayList;

public class Neuron {
    int aktFkt;
    ArrayList<Double> weights;
    int inputNum;

    public int getAktFkt() {
        return aktFkt;
    }

    public void setAktFkt(int aktFkt) {
        this.aktFkt = aktFkt;
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public void setWeights(int input, double weights) {
        this.weights.set(input, weights);
    }

    //input number damit weights direkt beim erstellen zugewiesen werden können
    public Neuron(int aktFkt, int inputNum) {
        this.aktFkt = aktFkt;
        weights = new ArrayList<>();
        this.inputNum = inputNum;
        asignrandomWeights();
    }

    //Das ist wahrscheinlich temp bis wir mit dem eigentlichen
    //lernen anfangen
    public void asignrandomWeights() {
        for (int i = 0; i <= inputNum; i++) {
            weights.add(Math.random());
        }
    }

    //Berechnet den Output des Knoten
    public double outputFkt(ArrayList<Double> input) {
        double sum = 0;
        //es wird in der Formel Bias gebraucht, keine Ahnung ob
        //wir den schon jetz brauchen
        double bias = 0;
        for (int j = 0; j < input.size(); j++) {
            sum += input.get(j) * weights.get(j);
        }
        sum += bias;
        return Aktivierungsfunktionen.funktionSelect(sum, aktFkt);
    }
}
