public class Aktivierungsfunktionen {
    public static double funktionSelect(double x, int aktFkt) {
        if (aktFkt == 0) {
            return identitaetsFunktion(x);
        } else if (aktFkt == 1) {
            //Weiß nicht genau wie die step Pos ausgewählt werden soll
            //Temporär
            return stepFunktion(x, 0);
        } else {
            return sigmoidFunktion(x);
        }
    }

    //Gibt den Eingabewert zurück
    public static double identitaetsFunktion(double x) {
        return x;
    }

    //Step Funktion macht einen Step bei StepXPos, also für x kleiner StepXPos -1, für x größer StepXPos 1, also StepXPos gibt an welche Step Funktion es ist
    public static double stepFunktion(double x, double stepXPos) {
        if (x < stepXPos) {
            return -1.0;
        } else {
            return 1.0;
        }
    }

    //
    public static double sigmoidFunktion(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
