import java.util.ArrayList;

public class Neuron {
    int aktFkt;
    ArrayList<Double> weights;
    public Neuron(int aktFkt){
        this.aktFkt = aktFkt;
    }

    public double outputFkt(ArrayList<Double> input){
        for(int i = 0; i <= input.size(); i++){
            weights.add(Math.random());
        }
        double sum = 0;
        for(int j = 0; j< input.size();j++){
            sum+= weights.get(j)* weights.get(j);
        }
        return sum;
    }
}
