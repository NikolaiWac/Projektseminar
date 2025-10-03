public class Aktivierungsfunktionen {
    public static double identitaetsFunktion(double x) {
        return x;
    }

    public static double stepFunktion(double x, double stepXPos){
            if (x < stepXPos) {
                return -1.0;
            }else  {
                return 1.0;
            }
    }

    public static double sigmoidFunktion(double x) {
        return 1.0/(1.0 + Math.exp(-x));
    }
}
