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

#include "multilayer_perceptron.h"

namespace Sys
{
	namespace ArtificialIntelligence
	{
		namespace EagerLearning
		{
			/**
			* creates a new instance of a PerceptronLayer object which represents the input layer
			* @param size			- number of neurons in the layer
			*/
			MultilayerPerceptron::PerceptronLayer::PerceptronLayer(int size) // used for input layer
			{
				//Initialize input layer
				if (size < 0)
				{
					throw std::invalid_argument("Parameter size cannot be less than 0");
				}
				size++; //+1 for bias
				this->upper = NULL;
				this->lower = NULL;
				this->learningRate = 0.0f;
				this->hasWeights = false;
				this->size = size; 
				data.resize(size, 0.0f);
				//Initialize pointers
				this->result = &data[0];
				this->error  = NULL;
				this->result[size -1] = -1.0f; //set bias to -1
				this->isOutput = false;
			}
			/**
			* creates a new instance of a PerceptronLayer object which represents a output or hidden layer
			* @param isOutput		- true for output layer, false for hidden layer
			* @param size			- number of neurons in the layer
			* @param learningRate   - float point value between 0.0 and 1.0
			* @param upper			- pointer to the upper layer
			*/
			MultilayerPerceptron::PerceptronLayer::PerceptronLayer(bool isOutput, int size, float learningRate, PerceptronLayer* upper) //used for output and hidden layer
			{
				//Initialize hidden or output layer	
				if (upper == NULL)
				{
					throw std::invalid_argument("Parameter upper cannot be NULL");
				}
				if (isOutput)
				{
					if (size < 1)
					{
						throw std::invalid_argument("Parameter size cannot less than 1");
					}
				}
				else
				{
					if (size < 0)
					{
						throw std::invalid_argument("Parameter size cannot less than 0");
					}
					size++; //+1 for bias
				}

				if (learningRate < 0.0f){learningRate = 0.0f;}
				if (learningRate > 1.0f){learningRate = 1.0f;}
				this->hasWeights = true;
				this->size = size;
				this->lower = NULL;
				this->upper = upper;
				this->learningRate = learningRate;
				this->isOutput = isOutput;

				//allocate memory for result, error vector and weight matrix in one shared float buffer and initialize to zero
				data.resize(size + size + size * upper->getSize(), 0.0f);

				//Create 2 dimensional weights array with [size, upper->getSize()] as dimension
				weights.resize(size, NULL); //Initialize with NULL just to be sure...

				//Initialize pointers
				this->result = &data[0];
				this->error  = &data[size];
				for (int i = 0; i < size; i++)
				{
					weights[i] = &data[2 * size + i * upper->getSize()];
				}
				//Initialize weights with random data (from -0.3 to +0.3) 
				for (int j = 0; j < upper->getSize(); j++)
				{
					for (int i = 0; i < size; i++)
					{
						weights[i][j] = float(rand() % 600) * 0.001f - 0.3f; //return value of rand() is guaranteed to be at least between 0 and 32767
					}
				}
				if (isOutput == false) // set bias to -1 (will not be modified later)
				{
					this->result[size - 1] = -1.0f; //set bias to -1
				}
			}
			/**
			* Frees all reserved ressources
			*/
			MultilayerPerceptron::PerceptronLayer::~PerceptronLayer()
			{
				//Nothing to do here for now...
			}
			/**
			* sets the values of all output neurons (except bias)
			* @param src        - pointer to an array with the float values to set
			* @return			- none
			*/
			void MultilayerPerceptron::PerceptronLayer::setResult(const float* src)
			{
				float *dst = result;
				if (isOutput)
				{
					//although this method will not be called for the output layer we should make it work correctly
					for (int i = 0; i < size; i++) // there is not bias value for the output layer!
						(*dst++) = (*src++);
				}
				else
				{
					for (int i = 0; i < size - 1; i++) // size-1 because we don't want to overwrite bias!
						(*dst++) = (*src++);
				}
			}
			/**
			* returns a pointer to the result calculated by a previous call to calculate()
			* @return		- float pointer to the result of the first neuron in the layer
			*/
			const float* MultilayerPerceptron::PerceptronLayer::getResult()
			{
				return result;
			}
			/**
			* Returns the number of neurons of the layer (including bias)
			* @return                      - number of neurons
			*/
			int MultilayerPerceptron::PerceptronLayer::getSize()
			{
				return size;
			}
			/**
			* Changes the learning rate of the layer
			* @param learningRate          - float point value between 0.0 and 1.0
			* @return                      - old learning rate
			*/
			float MultilayerPerceptron::PerceptronLayer::setLearningRate(float learningRate)
			{
				float oldLearningRate = this->learningRate;
				if (learningRate < 0.0f){learningRate = 0.0f;}
				if (learningRate > 1.0f){learningRate = 1.0f;}
				this->learningRate = learningRate;
				return oldLearningRate;
			}
			/**
			* Connects the layer with the upper and the lower layer
			* @param upper	- upper layer which is processed before this layer
			* @param lower	- lower layer which is processed after this layer
			* @return		- none
			*/
			void MultilayerPerceptron::PerceptronLayer::connect(PerceptronLayer* upper, PerceptronLayer* lower)
			{
				this->upper = upper;
				this->lower = lower;
			}
			/**
			* Calculatees the output of the layer
			* @return		- none
			*/
			void MultilayerPerceptron::PerceptronLayer::calculate()
			{

				if (hasWeights && (upper != NULL))
				{
					int upperSize = upper->getSize();
					for (int j = 0; j < size; j++)
					{
						float res = 0.0f;
						const float *upperResult = upper->getResult(); // (re)set pointer to result of upper layer
						float *row = weights[j];
						/*
						for (int i = 0; i <upperSize; i++) 
						{	
						res += (*upperResult++) * (*row++);
						}*/
						//Unrolling although it seems to improve performance not much...
						//making sure code is compiled for SSE2 architecture also improves performance
						for (int i = 0; i < (upperSize >> 2); i++)
						{		
							//Output of upper layer is input of the current layer
							res += upperResult[0] * row[0] //Faster as res += (*upperResult++) * (*row++); 
							+  upperResult[1] * row[1]
							+  upperResult[2] * row[2]
							+  upperResult[3] * row[3];
							row += 4;
							upperResult += 4;
						}
						//the rest (doing this at the end seems to be better for speed)
						switch (upperSize & 3)  // equal to upperSize % 4
						{
							res += (*upperResult++) * (*row++);
						case 3:
							res += (*upperResult++) * (*row++);
						case 2:
							res += (*upperResult++) * (*row++);
						case 1:
							res += (*upperResult++) * (*row++); //No ++ for last one needed, but who cares...
						case 0:
							break;
						}
						result[j] = sigmoid(res);
					}
				}
				/* unoptimized version
				for (int j = 0; j < size; j++)
				{
				float res = 0.0f;
				for (int i = 0; i < upper->getSize(); i++)
				{		
				res += upper->getResult()[i] * weights[j][i]; //Output of upper layer is input of the current layer
				}
				result[j] = sigmoid(res);
				}
				*/
			}
			/**
			* Update weights of the layer
			* @return		- none
			*/
			void MultilayerPerceptron::PerceptronLayer::updateWeights()
			{
				if (hasWeights && (upper != NULL))
				{
					//update weights with Backpropagation
					int upperSize = upper->getSize();
					for (int j=0; j < size; j++)
					{
						float d = learningRate * error[j];
						const float *upperResult = upper->getResult(); // (re)set pointer to result of upper layer, which is the input of this layer
						float *row = weights[j];
						for (int i=0; i < upperSize; i++)
						{
							(*row++) += d * (*upperResult++);  //(learning rate) * error * input
						}
					}
					/* unoptimized version
					//update weights
					for (int j=0; j < size; j++)
					{
					for (int i=0; i < upper->getSize(); i++)
					{
					weights[j][i] += learningRate * error[j] * upper->getResult()[i];  //(learning rate) * error * input
					}
					}
					*/
					
				}
			}
			/**
			* Calculates the error of the output layer
			* @param target	- pointer to an array with the desired output values
			* @return		- none
			*/
			void MultilayerPerceptron::PerceptronLayer::feedback(const float* target)
			{
				if (hasWeights) //If it has weights it also has an error vector...
				{
					float *ptrError = error;
					float *ptrResult = result;
					for (int i=0; i < size; i++) 
					{
						//The well known Hebbian learning rule
						float res = *ptrResult++;
						(*ptrError++) =  res * (1.0f - res) * ((*target++) - res);  // use (1.0f - result) because of missing symmetry between 0 and 1 in the Hebbian learning rule
					}
				}
				/* unoptimized version
				for (int i=0; i < size; i++)
				{
				//The well known Hebbian learning rule
				error[i] =  result[i] * (1.0f - result[i]) * (target[i] - result[i]);  // use (1.0f - result[i]) because of missing symmetry between 0 and 1 in the Hebbian learning rule
				}
				*/
			}
			/**
			* Prpagates the error to next higher layer
			* @return	- none
			*/
			void MultilayerPerceptron::PerceptronLayer::propagate()
			{
				if (hasWeights  && (lower != NULL)) //If it has weights it also has an error vector...
				{
					int lowerSize = lower->getSize();
					for (int j = 0; j < size; j++)
					{
						float err = 0.0f;
						float res = result[j];
						const float *lowerError = lower->error;
						float *lowerWeights = lower->weights[0] + j;
						for (int i=0; i < lowerSize; i++)
						{
							err += (*lowerError++) * (*lowerWeights);
							lowerWeights += lowerSize;
						}
						error[j] = err * res * (1.0f - res); 
					}
				}
				/* unoptimized version
				for (int j = 0; j < size; j++)
				{
				float err = 0.0f;
				for (int i=0; i < lower->getSize(); i++)
				{
				err += lower->error[i] * lower->weights[i][j];
				}
				error[j] = err * result[j] * (1.0f - result[j]); 
				}
				*/
			}
			/**
			* Sigmoid function which is used as activation function
			* @param x	- float value
			* @return	- float value
			*/
			float MultilayerPerceptron::PerceptronLayer::sigmoid(float x)
			{
				return 1.0f / (1.0f + expf(-x));
			}
			/**
			* Saturation function which is used as activation function
			* @param x	- float value
			* @return	- float value
			*/
			float MultilayerPerceptron::PerceptronLayer::saturate(float x)
			{
				if (x > 1.0f)
					x = 1.0f;
				if (x < 0.0f)
					x = 0.0f;
				return x;
			}
			/**
			* write data to the declared stream
			* @param file				    - stream where the data of the PerceptronLayer should be stored
			* @return						- none
			*/
			void MultilayerPerceptron::PerceptronLayer::writeStream(ofstream& file)
			{
				file.write((char*)&data[0], sizeof(float) * data.size());
			}
			/**
			* read data from the declared stream
			* @param file				    - stream which contains the data of the PerceptronLayer
			* @return						- none
			*/
			void MultilayerPerceptron::PerceptronLayer::readStream(ifstream& file)
			{
				file.read((char*)&data[0], sizeof(float) * data.size());
			}
			/**
			* Create a instance of a PerceptronLayers object
			* @param numLayers				- number of layers
			* @return						- none
			*/
			MultilayerPerceptron::PerceptronLayers::PerceptronLayers(int numLayers)
			{
				//Make sure the vector must not be resized later and is initialized with NULL pointers!
				layers.resize(numLayers, NULL);
			}
			/**
			* Frees all layers
			*/
			MultilayerPerceptron::PerceptronLayers::~PerceptronLayers()
			{
				for (unsigned int i = 0; i < layers.size(); i++)
				{
					if (layers[i] != NULL)
						delete layers[i];
				}
			}
			/**
			* Returns the number of leayers
			* @return						- number of layers
			*/
			int MultilayerPerceptron::PerceptronLayers::size()
			{
				return layers.size();
			}
			/**
			* Provides random access to all layers
			* @param idx					- index of the layer (0 is the input layer)
			* @throw std::invalid_argument	- if an invalid index is used
			* @return						- reference to a PerceptronLayer pointer
			*/
			MultilayerPerceptron::PerceptronLayer*& MultilayerPerceptron::PerceptronLayers::operator[](int idx)
			{
				if ((idx >= 0) && (idx < layers.size()) )
				{
					return layers[idx];
				}
				else
				{
					throw std::invalid_argument("illeagl PerceptronLayer index");
				}
			}
			/**
			* creates all layers of the multilayer perceptron
			* @param inputs                - The number of input neurons
			* @param outputs               - The number of output neurons
			* @param hidden                - The number of hidden neurons for multiple hidden layers
			* @param recurrent             - The number of recurrent neurons (neurons added to the last hidden layer that are connected with the input layer)
			* @param learningRate          - float point value between 0.0 and 1.0
			* @throw std::invalid_argument - if an invalid size parameter is declared
			* @throw std::bad_alloc        - if memory allcation fails
			* @return					   - none
			*/
			void MultilayerPerceptron::PerceptronLayers::createLayers(int inputs, int outputs, vector<int>& hidden, int recurrent, float learningRate)
			{
				int numLayers = hidden.size() + 2;
				if (recurrent < 0)
				{
					//Make sure number of recurrent neurons is not negative
					recurrent = 0;
				}
				PerceptronLayer* inputLayer = new PerceptronLayer(inputs + recurrent); //Input neurons consists of input-, recurrent neurons and one bias neuron
				PerceptronLayer* outputLayer = NULL;
				PerceptronLayer* last = inputLayer;

				//add input, hidden and output layers
				layers[0] = inputLayer;
				for (unsigned int i = 0; i < hidden.size(); i++)
				{
					PerceptronLayer* layer = NULL;
					if (i == (hidden.size() - 1))
					{
						// the last hidden layer
						layer = new PerceptronLayer(false, hidden[i] + recurrent, learningRate, last);
					}
					else
					{
						layer = new PerceptronLayer(false, hidden[i], learningRate, last);
					}

					last = layer;
					layers[i + 1] = layer; //+1 because we should not overwrite input layer!
				}
				outputLayer = new PerceptronLayer(true, outputs, learningRate, last);
				layers[numLayers - 1] = outputLayer;

				//connect layers
				inputLayer->connect(NULL, layers[1]);
				outputLayer->connect(layers[numLayers-2], NULL);

				for (int i = 1; i < numLayers-1; i++)
				{
					layers[i]->connect(layers[i - 1], layers[i + 1]);
				}
			}
			/**
			* Creates a new instance of a multilayer Perceptron.
			* All weigts are initialized with random values between -0.3 to + 0.3.
			* @param inputs                - The number of input neurons
			* @param outputs               - The number of output neurons
			* @param hidden                - The number of hidden neurons for multiple hidden layers
			* @param recurrent             - The number of recurrent neurons (neurons added to the last hidden layer that are connected with the input layer)
			* @param learningRate          - float point value between 0.0 and 1.0
			* @throw std::invalid_argument - if an invalid size parameter is declared
			* @throw std::bad_alloc        - if memory allcation fails
			*/
			MultilayerPerceptron::MultilayerPerceptron(int inputs, int outputs, vector<int>& hidden, int recurrent, float learningRate) : layers(2 + hidden.size())
			{
				if (hidden.size() == 0) //recurrent neurons are only possible if we have at least one hidden layer
				{
					recurrent = 0;
				}
				layers.createLayers(inputs, outputs, hidden, recurrent, learningRate);
				numLayers = layers.size();
				inputLayer = layers[0];
				outputLayer = layers[numLayers - 1];
				//Needed for recurrent neurons
				inputData.resize(inputs + recurrent, 0.0f);
				numInputs = inputs;
				numRecurrent = recurrent;

				lastHiddenLayer = NULL;
				if (numLayers > 2) // if there is a hidden layer
				{
					lastHiddenLayer = layers[numLayers - 2]; // the last hidden layer
				}
			}
			/**
			* Creates a new instance of a single layer Perceptron.
			* All weigts are initialized with random values between -0.3 to + 0.3.
			* @param inputs                - The number of input neurons
			* @param outputs               - The number of output neurons
			* @param learningRate          - float point value between 0.0 and 1.0
			* @throw std::invalid_argument - if an invalid size parameter is declared
			* @throw std::bad_alloc        - if memory allcation fails
			*/
			MultilayerPerceptron::MultilayerPerceptron(int inputs, int outputs, float learningRate) : layers(2)
			{
				vector<int> hidden;
				layers.createLayers(inputs, outputs, hidden, 0, learningRate);
				numLayers = layers.size();
				inputLayer = layers[0];
				outputLayer = layers[numLayers - 1];
				//Needed for recurrent neurons
				inputData.resize(inputs, 0.0f);
				numInputs = inputs;
				numRecurrent = 0;

				lastHiddenLayer = NULL;
				if (numLayers > 2) // if there is a hidden layer
				{
					lastHiddenLayer = layers[numLayers - 2]; // the last hidden layer
				}
			}
			/**
			* Frees all reserved ressources
			*/
			MultilayerPerceptron::~MultilayerPerceptron()
			{
				//Nothing to do here...
			}
			/**
			* calculates the output of the multilayer perceptron based on the declared input and weights
			* @param input		- input vector with the dimension declared in the constructor
			* @return			- none
			*/
			void MultilayerPerceptron::calculate(const float* input)
			{
				//build input with recurrent neurons
				for (int i = 0; i < numInputs; i++)
				{
					inputData[i] = input[i];
				}
				if (lastHiddenLayer != NULL)
				{
					const float *result = lastHiddenLayer->getResult();
					for (int i = 0; i < numRecurrent; i++)
					{
						inputData[numInputs + i] = result[i]; 
					}
				}

				inputLayer->setResult(&inputData[0]); //set input vector
				for (int i = 1; i < numLayers; i++) //for all hidden layers and the output layer
				{
					layers[i]->calculate();
				}
			}
			/**
			* returns a pointer to the result of the previous call to calculate()
			* @return		- float pointer to the result of the first output neuron
			*/
			const float* MultilayerPerceptron::getResult()
			{
				return outputLayer->getResult(); //Get result of output layer
			}
			/**
			* adapts the weights of the mulitlayer perceptron
			* @param target                - value which should have been returned by calculate()
			* @return                      - none
			*/
			void MultilayerPerceptron::feedback(const float* target)
			{
				outputLayer->feedback(target); //output layer
				for (int i = numLayers - 2; i > 0; i--)
				{	
					layers[i]->propagate();
				}
				//Update weights 
				for (int i = 1; i < numLayers; i++) //all layers except input layer
				{
					layers[i]->updateWeights();
				}
			}
			/**
			* Changes the learning rate for all layers of the multilayer perceptron
			* @param learningRate          - float point value between 0.0 and 1.0
			* @return                      - none
			*/
			void MultilayerPerceptron::setLearningRate(float learningRate)
			{
				for (int i = 1; i < numLayers; i++) //for all hidden layers and the output layer
				{
					layers[i]->setLearningRate(learningRate);
				}
			}
			/**
			* Loads weights of all layers of the multilayer perceptron from an binary file
			* @param *fileName			- Path to file which should be opened
			* @throw ios_base::failure - if opening or writing the file fails
			* @return					- none
			*/
			void MultilayerPerceptron::loadFile(const char *fileName)
			{
				char magic[4];
				ifstream file;
				file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
				file.open(fileName, ios::in | ios::binary);
				//read magic
				file.read(magic, 4);
				//magic must fit
				if ((magic[0] == 'M') && (magic[1] == 'L') && (magic[2] == 'P') && (magic[3] == '1'))
				{
					for (int i = 0; i < numLayers; i++) //all layers
					{
						layers[i]->readStream(file);
					}
				}
				else
				{
					throw ios_base::failure("invalid file header (MLP1 expected)");
				}
				file.close();
			}
			/**
			* Saves weights of all layers of the multilayer perceptron to a binary file.
			* If the file already exits it will be overwritten.
			* @param *fileName         - Path to file which should be created
			* @throw ios_base::failure - if opening or writing the file fails
			* @return                  - none
			*/
			void MultilayerPerceptron::saveFile(const char *fileName)
			{
				ofstream file;
				try
				{
					file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
					file.open(fileName, ios::out | ios::binary);
					char magic[4] = {'M','L','P', '1'};	
					//write header
					file.write(magic, 4);

					for (int i = 0; i < numLayers; i++) //all layers
					{
						layers[i]->writeStream(file);
					}
					file.close();
				}
				catch(...)
				{
					//delete partly created file
					file.exceptions(ios_base::goodbit); //make sure no further exception will be thrown
					file.close(); // first close file
					remove(fileName);
					throw; // throw exception again
				}
			}

		}
	}
}
