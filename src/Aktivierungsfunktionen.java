public class Aktivierungsfunktionen {
    public static double funktionSelect(double x, int aktFkt){
        if(aktFkt == 0){
            return identitaetsFunktion(x);
        }else if(aktFkt == 1){
            //Weiß nicht genau wie die step Pos ausgewählt werden soll
            //Temporär
            return stepFunktion(x, 0);
        }else{
            return sigmoidFunktion(x);
        }
    }

    //0
    public static double identitaetsFunktion(double x) {
        return x;
    }

    //1
    public static double stepFunktion(double x, double stepXPos){
            if (x < stepXPos) {
                return -1.0;
            }else  {
                return 1.0;
            }
    }

    //2
    public static double sigmoidFunktion(double x) {
        return 1.0/(1.0 + Math.exp(-x));
    }
}
