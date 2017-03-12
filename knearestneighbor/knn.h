/**
 * @brief k-Nearest Neighbor Algorithm
 *
 * class which implements the k-Nearest Neighbor algorithm
 *
 * Author:	Stefan Moebius (mail@stefanmoebius.de)
 *
 * Date:	2011-05-19
 *
 * Licence: Released to the PUBLIC DOMAIN
 *
 * Usage:
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 */

#ifndef _KNEARESTNEIGHBOR_H_
#define _KNEARESTNEIGHBOR_H_

#define _USE_STD_VECTOR_CLASS
//#define _USE_GAUSSIAN_DISTRIBUTION

#include <float.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string.h>

//Visual C++ does not support architecture independant "int32_t" data type before VS 2010
#ifdef _MSC_VER
typedef int int32_t; // int32_t is declared as "int" and not as "__int32"
#else
#include <stdint.h>
#endif

#ifdef _USE_STD_VECTOR_CLASS
/* contiguously memory is expected (&v[x] is used)
"The elements of a vector are stored contiguously, meaning that if v is a vector<T, Allocator> where T is some type other than bool, then it obeys the identity &v[n] == &v[0] + n for all 0 <= n < v.size()." 
see ISO 14882, 2nd ed., 23.2.4 [lib.vector]:
http://cs.nyu.edu/courses/summer11/G22.2110-001/documents/c++2003std.pdf 
*/
#include <vector>
//typedef vector floatVector;
#define floatVector vector<float>
#else
#include "../../Array/rawVector.h"
using namespace Sys::Array;
//typedef rawVector floatVector 
#define floatVector rawVector<float>
#endif

#ifndef M_SQRT_PI
#define M_SQRT_PI    1.7724538509055159f
#endif

#ifndef M_SQRT_2
#define M_SQRT_2    1.4142135623730951f
#endif

using namespace std;

namespace Sys
{
	namespace ArtificialIntelligence
	{
		namespace LazyLearning
		{
			/**
			* Implentation of the k-Nearest Neighbor algorithm
			* Features:
			* - multi-dimensional input and output vectors
			* - optimized training
			* - export/import training data as/from ASCII file
			* - save/load training data to binary file (fast)
			* @see http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
			*/
			class KNearestNeighbor
			{
				friend class KNearestNeighborHashed;
				/**
				* Helper class of KNearestNeighbor, do not use seperatly
				*/
				class Neighbors
				{
					friend class KNearestNeighbor;
				protected: //all methods protected, so we can access it only from KNearestNeighbor
					Neighbors(int numNeighbors);
					~Neighbors();
					inline void insertNeighbor(int idx, float dist);
					inline float getWorstDistance();
				private:
					struct NEIGHBOR
					{
						int idx; //index of sample
						float distQ; //quadratic distance
					};
					NEIGHBOR* pointer;
					int numNeighbors;

					//disallow copy and assign
					Neighbors(const Neighbors&);               
					void operator=(const Neighbors&);
				};

			public:
				//Methods:
				KNearestNeighbor(int dimIn, int dimOut);
				~KNearestNeighbor();
				void clear();
				void exportFile(const char *fileName);
				void importFile(const char *fileName);
				void saveFile(const char *fileName);
				void loadFile(const char *fileName, bool append);
				void calculate(const float *vecInput, float *vecOutput, int numNeighbors = 1);
				void train(const float *vecInput, const float *vecOutput, float weight = 1.0f, float alpha = 1.0f);
				void optimizeTraining(bool enable, float maxError, float maxDistance, int maxSamples = 0);
				int getNumberOfSamples();
			protected:
				void append(const float *vecInput, const float *vecOutput, float weight = 1.0f, float alpha = 1.0f);
				void writeStream(ofstream *file);
				void readStream(ifstream *file, bool append);
				//Vector helper function(s)
				inline float vecDistQ(const float *vecA, const float *vecB, int dim);
				//C-Style implentation of kNN algorithm
				void kNN(const float *input, int dimension, const float *samples, int numSamples, Neighbors *neighbors);
				float kNNOutput(const float *sampleOutput, int dimOutput, int idxOutput, const float *sampleWeight, const float *sampleAlpha, int numSamples, Neighbors *neighbors);
				//Members:
				int dimIn;  //Input dimensions
				int dimOut; //Output dimensions
				int numSamples;
				floatVector sampleInput;
				floatVector sampleOutput;
				floatVector sampleAlpha;
				floatVector sampleWeight;
				//Optimized training
				bool enableOptimize;
				float maxError, maxDistance;
				int maxSamples;
			private:
				//disallow copy and assign
				KNearestNeighbor(const KNearestNeighbor&);               
				void operator=(const KNearestNeighbor&);
			};
		}
	}
}
#endif

/* Sample code:

#include "Sys/ArtificialIntelligence/LazyLearning/kNN.h"
#include <ostream>
#include <conio.h>

using namespace Sys::ArtificialIntelligence::LazyLearning;
...

KNearestNeighbor nn(3, 1); //Use 3 dimensional vectors as input and scalar (1 dimension) as output

//Add first sample vector
float sampleInput[3];
float sampleOutput = 1.0f;
sampleInput[0] = 1.0f;
sampleInput[1] = 2.0f;
sampleInput[2] = 3.0f;
nn.train(sampleInput, &sampleOutput);

//Add second sample vector
sampleInput[0] = 4.0f;
sampleInput[1] = 4.0f;
sampleInput[2] = 4.0f;
sampleOutput = 0.5f;
nn.train(sampleInput, &sampleOutput);

//Check the output for an input vector
float testInput[3];
float testOutput;
testInput[0] = 1.0f;
testInput[1] = 0.0f;
testInput[2] = 1.0f;

nn.calculate(testInput, &testOutput, 2);
cout << testOutput << endl;

nn.saveFile("knn.dat");
getch();
...
*/
