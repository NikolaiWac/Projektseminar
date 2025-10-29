import java.util.ArrayList;

public class Aktivierungsfunktionen {

    //für funktionen die mehr infos brauchen um zu funktionieren, einfache funktionen sollten trotzdem noch funktionieren
    public static double funktionSelect(double x, int aktFkt, ArrayList<Double> furtherInfo) {
        if (aktFkt == 0) {
            return identitaetsFunktion(x);
        } else if (aktFkt == 1) {
            if (furtherInfo != null) {
                return stepFunktion(x, furtherInfo.get(0), furtherInfo.get(1), furtherInfo.get(2));
            }
            else {
                return stepFunktion(x);
            }
        } else {
            return sigmoidFunktion(x);
        }
    }

    //Gibt den Eingabewert zurück
    public static double identitaetsFunktion(double x) {
        return x;
    }

    //vordefinierte Step Funktion mit hardcoded values
    public static double stepFunktion(double x) {
        if (x < 1) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    //Zweite Step- Funktion die user noch mehr Möglichkeiten erlaubt
    public static double stepFunktion(double x, double stepXPos, double leftVal, double rightVal) {
        if (x < stepXPos) {
            return leftVal;
        } else {
            return rightVal;
        }
    }

    //
    public static double sigmoidFunktion(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
