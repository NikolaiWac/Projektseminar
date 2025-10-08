import java.util.ArrayList;

public class Neuron {
    int aktFkt;
    ArrayList<Double> weights;

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

    public Neuron(int aktFkt) {
        this.aktFkt = aktFkt;
        weights = new ArrayList<>();
    }

    //Das ist wahrscheinlich temp bis wir mit dem eigentlichen
    //lernen anfangen
    public void asignrandomWeights(int input) {
        for (int i = 0; i <= input; i++) {
            weights.add(Math.random());
        }
    }

    //Berechnet den Output des Knoten
    public double outputFkt(ArrayList<Double> input) {
        asignrandomWeights(input.size());
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
