#include <iostream>
#include "knn.h"

using namespace std;
using namespace Sys::ArtificialIntelligence::LazyLearning;

#define DIM_INPUT 2
#define DIM_OUTPUT 1

int main()
{
	float input[DIM_INPUT];
	float target[DIM_OUTPUT];
	float output[DIM_OUTPUT];

	KNearestNeighbor knn(DIM_INPUT, DIM_OUTPUT);

	//knn.LoadFile("knn.dat", false); // gespeicherte Datei laden
	for (int train = 0; train < 10000; train++)
	{
		//Eingabewerte
		input[0] = (rand() % 1000)/1000.0;
		input[1] = (rand() % 1000)/1000.0;

		//Sollwert
		target[0] = (input[0] + input[1]) * 0.5f; //Ziel ist der Mittelwert

		//Ausgabe berechen
		knn.calculate(input, output, 5); //Nächste 5 Nachbarn berücksichtigen

		cout << "Ausgabe: "<< output[0] << "  Soll: " << target[0] <<endl; 
		cout << "Fehler: " << fabs(output[0] - target[0]) << endl;

		//Training
		knn.train(input, target);
	}
	//knn.SaveFile("knn.dat"); // gelerntes abspeichern
	
	//Auf Eingabe Warten
	int dummy;
	cin >> dummy;
}