#include <vector>
#include <iostream>
#include "multilayer_perceptron.h"

using namespace std;
using namespace Sys::ArtificialIntelligence::EagerLearning;

#define DIM_INPUT 2
#define DIM_OUTPUT 1
#define LEARNINGFACTOR 0.1

int main()
{
	float input[DIM_INPUT];
	float target[DIM_OUTPUT];

	MultilayerPerceptron per(DIM_INPUT, DIM_OUTPUT, LEARNINGFACTOR);  //Neuronales Netzt erstellen

	//per.loadFile("perceptron.dat"); //Gewichte aus Datei laden (muss zuvor mit SaveFile() abgespeichert werden)

	for (int train = 0; train < 10000; train++)
	{
		//Eingabewerte
		input[0] = (rand() % 1000)/1000.0;
		input[1] = (rand() % 1000)/1000.0;

		//Sollwert
		target[0] = (input[0] + input[1]) * 0.5f; //Ziel ist Mittelwert

		//Ausgabe berechen
		per.calculate(input);

		cout << "Ausgabe: "<< per.getResult()[0] << "  Soll: " << target[0] <<endl; 
		cout << "Fehler: " << fabs(per.getResult()[0] - target[0]) << endl;

		//Training
		per.feedback(target); //wichtig ist, dass vor feedback() immer calculate() aufgerufen wird
	}
	per.saveFile("perceptron.dat"); // Gewichte speichern

	//Auf Eingabe Warten
	int dummy;
	cin >> dummy;	
}