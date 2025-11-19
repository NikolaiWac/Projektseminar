public class ActFuntions {

    //für funktionen die mehr infos brauchen um zu funktionieren, einfache funktionen sollten trotzdem noch funktionieren
    public static double funkcionSelect(double x, int aktFkt, double[] furtherInfo) {
        if (aktFkt == 0) {
            return identityFunction(x);
        } else if (aktFkt == 1) {
            if (furtherInfo != null) {
                return stepFunction(x, furtherInfo[0], furtherInfo[1], furtherInfo[2]);
            }
            else {
                return stepFunction(x);
            }
        } else {
            return sigmoidFunction(x);
        }
    }

    //Gibt den Eingabewert zurück
    public static double identityFunction(double x) {
        return x;
    }

    //vordefinierte Step Funktion mit hardcoded values
    public static double stepFunction(double x) {
        if (x < 1) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    //Zweite Step- Funktion die user noch mehr Möglichkeiten erlaubt
    public static double stepFunction(double x, double stepXPos, double leftVal, double rightVal) {
        if (x < stepXPos) {
            return leftVal;
        } else {
            return rightVal;
        }
    }

    //
    public static double sigmoidFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double derivativeSelect(int aktFkt, double x) {
        if (aktFkt == 0) {
            return identityFunctionDerivation();
        } else if (aktFkt == 1) {
            return stepFunctionDerivation();
            }
        else {
            return sigmoidFunction(x);
        }
    }

    public static double identityFunctionDerivation() {
        return 1.0;
    }

    public static double stepFunctionDerivation() {
        return 1.0;
    }

    public static double sigmoidFunctionDerivation(double x) {
        return sigmoidFunction(x) * (1 - sigmoidFunction(x));
    }
}
