/**
* @brief Multilayer Perceptron Algorithm
*
* flexible implementation of a multilayer perceptron
*
*
* Author:	Stefan Moebius (mail@stefanmoebius.de)
*
* Date:	2011-08-07
*
* Licence: Released to the PUBLIC DOMAIN
*
* THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY
* KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
* PARTICULAR PURPOSE.
*/

#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdlib>

using namespace std;
namespace Sys
{
	namespace ArtificialIntelligence
	{
		namespace EagerLearning
		{
			class MultilayerPerceptron
			{
				class PerceptronLayer
				{
				public:
					PerceptronLayer(int size);
					PerceptronLayer(bool isOutput, int size, float learningRate, PerceptronLayer* upper);
					virtual ~PerceptronLayer();
					inline void setResult(const float* src);
					inline const float* getResult();
					inline int getSize();
					inline float setLearningRate(float learningRate);
					inline void connect(PerceptronLayer* upper, PerceptronLayer* lower);
					inline void calculate();
					void updateWeights();
					void feedback(const float* target);
					void propagate();
					void writeStream(ofstream& file);
					void readStream(ifstream& file);
				protected:
					inline static float sigmoid(float x);
					inline static float saturate(float x);
					int size;
					float learningRate;
					bool hasWeights, isOutput;
					PerceptronLayer *upper, *lower;
					float *result, *error;
					vector<float> data; //contains space for result, error and weights
					vector<float*> weights;
				private:
					//disallow copy and assign
					PerceptronLayer(const PerceptronLayer&);               
					void operator=(const PerceptronLayer&);
				};

				class PerceptronLayers
				{
				public:
					PerceptronLayers(int numLayers);
					virtual ~PerceptronLayers();
					inline int size();
					inline PerceptronLayer*& operator[](int idx);
					void createLayers(int inputs, int outputs, vector<int>& hidden, int recurrent, float learningRate);
				private:
					vector<PerceptronLayer*> layers;
					//disallow copy and assign
					PerceptronLayers(const PerceptronLayers&);               
					void operator=(const PerceptronLayers&);
				};

			public:
				MultilayerPerceptron(int inputs, int outputs, vector<int>& hidden, int recurrent, float learningRate);
				MultilayerPerceptron(int inputs, int outputs, float learningRate);
				~MultilayerPerceptron();
				void calculate(const float* input);
				const float* getResult();
				void feedback(const float* target);
				void setLearningRate(float learningRate);
				void loadFile(const char *fileName);
				void saveFile(const char *fileName);
			protected:
				int numLayers;
				PerceptronLayers layers;
				PerceptronLayer *inputLayer;
				PerceptronLayer *outputLayer;
				//needed for recurrent neurons
				vector<float> inputData;
				int numInputs, numRecurrent;
				PerceptronLayer *lastHiddenLayer;
			private:
				//disallow copy and assign
				MultilayerPerceptron(const MultilayerPerceptron&);               
				void operator=(const MultilayerPerceptron&);
			};
		}
	}
}
#endif
