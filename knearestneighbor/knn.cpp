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

#include "knn.h"

namespace Sys
{
	namespace ArtificialIntelligence
	{
		namespace LazyLearning
		{
#pragma region "Public methods of class KNearestNeighbor"
			/**
			* Creates a new instance of KNearestNeighbor
			* @param dimIn                 - The dimension of input vectors (cannot be changed later)
			* @param dimOut                - The dimension of output vectors (cannot be changed later)
			* @throw std::invalid_argument - if an invalid parameter is declared (dimIn or dimOut less than 1)
			*/
			KNearestNeighbor::KNearestNeighbor(int dimIn, int dimOut)
			{
				if (dimIn < 1)
				{
					throw std::invalid_argument("dimIn cannot be less than 1");
				}
				if (dimOut < 1)
				{
					throw std::invalid_argument("dimOut cannot be less than 1");
				}
				this->numSamples = 0;
				this->dimIn = dimIn;
				this->dimOut = dimOut;

				//Optimized training
				enableOptimize = false; // disabled by default
				maxError = 0.0f;
				maxDistance = 0.0f;
				maxSamples = 0;
			}
			/**
			* Frees all reserved ressources
			*/
			KNearestNeighbor::~KNearestNeighbor()
			{
				this->dimIn = 0;
				this->dimOut = 0;
				this->numSamples = 0;
				clear();
			}
			/**
			* Removes all training data
			* @return -  none
			*/
			void KNearestNeighbor::clear()
			{
				this->numSamples = 0;
				this->sampleInput.clear();
				this->sampleOutput.clear();
				this->sampleWeight.clear();
				this->sampleAlpha.clear();
			}
			/**
			* Calculates an output vector for the declared input vector.
			* For the calculation the k-Nearest Neighbor algorithm is used.
			* It searches the k (declared in numNeighbors) nearest neighbor vectors for the input vector (*vecInput).
			* As metric the euclidean distance is used (vectors are considered as points here). 
			* The dimension of the input vector (*vecInput) and output vector (*vecOutput) must fit the dimensions declared in the constructor (dimIn and dimOut).
			* @param vecInput              - Pointer to an float array which is used as input. The number of elements must fit the delcared "dimIn" in the constructor.
			* @param vecOutput             - Pointer to an float array which will be filled with the output vector. The number of elements must fit the delcared "dimOut" in the constructor.
			* @param numNeighbors          - [Optional] How many neighbors are used for interpolation (1 by default)
			* @throw std::invalid_argument - if an invalid parameter is declared (NULL-pointer)
			* @throw std::bad_alloc        - if memory allcation fails
			* @return                      - none
			*/
			void KNearestNeighbor::calculate(const float *vecInput, float *vecOutput, int numNeighbors)
			{
				if (vecInput == NULL)
				{
					throw std::invalid_argument("vecInput cannot be null!");
				}
				if (vecOutput == NULL)
				{
					throw std::invalid_argument("vecOutput cannot be null!");
				}
				// correct bad/unuseful values for numNeighbors
				if (numNeighbors > numSamples)
				{
					numNeighbors = numSamples;
				}
				if (numNeighbors <= 0)
				{
					numNeighbors = 1;
				}
				if (numSamples > 0) //Needed becuase std::vector raises exception when count is 0
				{
					Neighbors nb = Neighbors(numNeighbors); // can throw std::bad_alloc(), or std::invalid_argument
					//Calculate nearest neighbors
					kNN(vecInput, dimIn, &sampleInput[0], numSamples, &nb);				
					//Calculate final output
					for (int i = 0; i < dimOut; i++)
					{
						vecOutput[i] = kNNOutput(&sampleOutput[0], dimOut, i, &sampleWeight[0], &sampleAlpha[0], numSamples, &nb);
					}
				}
				else
				{
					//Set output vector to 0 in this case...
					for (int i = 0; i < dimOut; i++)
					{
						vecOutput[i] = 0.0f;
					}
				}
			}
			/**
			* Adds a training set (input- and desired output vector)
			* @param vecInput              - Pointer to an input vector. The number of elements must fit the delcared "dimIn" in the constructor.  
			* @param vecOutput             - Desired output vector for the input vector. The number of elements must fit the delcared "dimOut" in the constructor. 
			* @param weight                - [Optional] strength of influence of the sample vector (1.0f by default)
			* @param alpha                 - [Optional] range of influence of the sample vector (1.0f by default)
			* @throw std::bad_alloc        - if memory allcation fails		 
			* @return                      - none
			*/
			void KNearestNeighbor::append(const float *vecInput, const float *vecOutput, float weight, float alpha)
			{
				//error checking only in public mehtod train()
#ifdef _USE_STD_VECTOR_CLASS
				sampleInput.resize((numSamples + 1) * dimIn);
				sampleOutput.resize((numSamples + 1) * dimOut);
				sampleWeight.resize(numSamples + 1);
				sampleAlpha.resize(numSamples + 1);
				memcpy(&sampleInput[numSamples * dimIn], &vecInput[0], sizeof(float) * dimIn);
				memcpy(&sampleOutput[numSamples * dimOut], &vecOutput[0], sizeof(float) * dimOut);
				sampleWeight[numSamples] = weight;
				sampleAlpha[numSamples] = alpha;
#else
				sampleInput.setVector(numSamples * dimIn, vecInput, dimIn);
				sampleOutput.setVector(numSamples * dimOut, vecOutput, dimOut);
				sampleWeight.setVector(numSamples, &weight, 1);
				sampleAlpha.setVector(numSamples, &alpha, 1);
#endif
				//Increase sample count only if ALL data is set correctly 
				numSamples++;
			}
			/**
			* Enables or disables optimized training(disabled by default).
			* If enabled the distance and the difference of the output to the nearest neighbor is checked.
			* If the distance is smaller than "maxDistance" or the relative difference of the output is smaller than "maxError" then no new training set is added.
			* This helps to make sure that the number of traing sets do not grow too fast. However, the train() method can execute substantially slower.
			* @param enable				   - enables or disables optimization (disabled by default)
			* @param maxError              - maximum tolerable relative error (in this case no new traning data is added, instead the old data gets updated)
			* @param maxDistance           - maximum tolerable distance the input vector can have to its nearest neighbor (otherwise a new traing set is added) 		 
			* @param maxSamples			   - maximum number of samples which can be added with train(). Use "0" if you don't want declare a maximum
			* @return                      - none
			*/
			void KNearestNeighbor::optimizeTraining(bool enable, float maxError, float maxDistance, int maxSamples)
			{
				this->enableOptimize = enable;
				this->maxError = maxError;
				this->maxDistance = maxDistance;
				this->maxSamples = maxSamples;
			}
			/**
			* returns the current number of samples added with the train() method
			* @return                      - number of added samples
			*/
			int KNearestNeighbor::getNumberOfSamples()
			{
				return this->numSamples;
			}
			/**
			* Adds a training set (input- and desired output vector)
			* @param vecInput              - Pointer to an input vector. The number of elements must fit the delcared "dimIn" in the constructor.  
			* @param vecOutput             - Desired output vector for the input vector. The number of elements must fit the delcared "dimOut" in the constructor. 
			* @param weight                - [Optional] strength of influence of the sample vector (1.0f by default)
			* @param alpha                 - [Optional] range of influence of the sample vector (1.0f by default)
			* @throw std::invalid_argument - if an invalid parameter is declared (NULL-pointer)
			* @throw std::bad_alloc        - if memory allcation fails		 
			* @return                      - none
			*/
			void KNearestNeighbor::train(const float *vecInput, const float *vecOutput, float weight, float alpha)
			{
				if (vecInput == NULL)
				{
					throw std::invalid_argument("vecInput cannot be null!");
				}
				if (vecOutput == NULL)
				{
					throw std::invalid_argument("vecOutput cannot be null!");
				}

				if (enableOptimize)
				{
					float error = 0.0f, dist = 0.0f;
					int idx = -1;
					bool canAddSamples = true;
					if (numSamples > 0) //Needed becuase std::vector raises exception when count is 0
					{
						Neighbors nb = Neighbors(1); // can throw std::bad_alloc(), or std::invalid_argument
						//Calculate nearest neighbors
						kNN(vecInput, dimIn, &sampleInput[0], numSamples, &nb);
						idx = nb.pointer->idx;
						dist = sqrtf(nb.pointer->distQ);
						//Determine the relative error of the component with the biggest difference
						for (int i = 0; i < dimOut; i++)
						{
							float result = kNNOutput(&sampleOutput[0], dimOut, i, &sampleWeight[0], &sampleAlpha[0], numSamples, &nb);
							float outValue = fabsf(vecOutput[i] - result);
							if (fabsf(result) > FLT_EPSILON)
							{
								error = max<float>(error, outValue / fabsf(result));
							}
						}
					}

					if ((numSamples >= maxSamples) && (maxSamples > 0))
					{
						canAddSamples = false;
					}

					if ( ((error <= maxError) || (dist <= maxDistance) || (canAddSamples == false)) && (idx >= 0))	// if error or distance is small...
					{
						//Just adjust existing data...
						float oldWeight = sampleWeight[idx];
						float resWeight = oldWeight + weight;
						sampleWeight[idx] = resWeight; // add the weight to the nearest neighbor
						//... and build the mean of both, input and output vectors
						if (fabsf(resWeight) > FLT_EPSILON)
						{
							sampleAlpha[idx] = (oldWeight * sampleAlpha[idx] + weight * alpha) / resWeight;
							for (int i = 0; i< dimIn; i++)
							{
								sampleInput[idx * dimIn + i] =  (oldWeight * sampleInput[idx * dimIn + i] + weight * vecInput[i]) / resWeight; 
							}
							for (int i = 0; i< dimOut; i++)
							{		
								sampleOutput[idx * dimOut + i] =  (oldWeight * sampleOutput[idx * dimOut + i] + weight * vecOutput[i]) / resWeight; 
							}
						}
					}
					else
					{
						//error to big or first traning set...
						if (canAddSamples)
							append(vecInput, vecOutput, weight, alpha);
					}
				}
				else
				{
					//unoptimized training
					append(vecInput, vecOutput, weight, alpha);
				}
			}
			/**
			* Writes the whole training data to an ASCII text file.
			* If the file already exits it will be overwritten.
			* @param *fileName         - Path to file which should be created
			* @throw ios_base::failure - if opening or writing the file fails		
			* @return                  - none
			*/
			void KNearestNeighbor::exportFile(const char *fileName)
			{
				ofstream file;
				//by default streams do not throw exceptions...
				file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
				try
				{
					file.open(fileName);
					file << dimIn << "\n";
					file << dimOut << "\n";
					file << numSamples << "\n";
					// write all samples into the file
					for (int i = 0; i < numSamples; i++)
					{
						// write all dimensions of the input vector
						for (int j = 0; j < dimIn; j++)
						{
							file << sampleInput[i * dimIn + j] << "\t";
						}
						// write all dimensions of the output vector
						for (int j = 0; j < dimOut; j++)
						{
							file << sampleOutput[i * dimOut + j] << "\t";
						}
						// write weight- and alpha-value for the sample vector
						float weight = sampleWeight[i], alpha = sampleAlpha[i];
						file << weight << "\t";
						file << alpha << "\t";
						file << "\n";
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
			/**
			* Append sample data from an ASCII text file
			* @param *fileName		   - Path to file which should be opened
			* @throw ios_base::failure - if opening or reading the file fails
			* @throw std::bad_alloc    - if memory allcation fails	
			* @return                  - none
			*/
			void KNearestNeighbor::importFile(const char *fileName)
			{
				int newNumSamples, newDimIn, newDimOut;
				float weight, alpha;
				bool result = false, failed = false;
				int offset = numSamples;
				ifstream file;
				// arrays to store vectors temporary (throws exception if it fails)
				floatVector vecIn(dimIn);
				floatVector vecOut(dimOut);

				file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
				file.open(fileName);
				file >> newDimIn;
				file >> newDimOut;
				file >> newNumSamples;
				if ((newDimIn == dimIn) && (newDimOut == dimOut))
				{
					if (newNumSamples > 0)
					{
						// read all samples in the file
						for (int i = 0; i < newNumSamples; i++)
						{
							// read all dimensions of the input vector
							for (int j = 0; j < newDimIn; j++)
							{
								file >> (vecIn[j]);
							}
							// read all dimensions of the output vector
							for (int j = 0; j < newDimOut; j++)
							{
								file >> (vecOut[j]);
							}
							//read weight- and alpha-value of the sample
							file >> weight;
							file >> alpha;
#ifdef _USE_STD_VECTOR_CLASS
							sampleInput.resize((offset + 1) * dimIn);
							sampleOutput.resize((offset + 1) * dimOut);
							sampleWeight.resize(offset + 1);
							sampleAlpha.resize(offset + 1);
							memcpy(&sampleInput[offset * dimIn], &vecIn[0], sizeof(float) * dimIn);
							memcpy(&sampleOutput[offset * dimOut], &vecOut[0], sizeof(float) * dimOut);
							sampleWeight[offset] = weight;
							sampleAlpha[offset] = alpha;
#else
							//store sample (std::bad_alloc is thrown if memory allocation fails)
							sampleInput.setVector(dimIn * offset, vecIn.getPointer(), dimIn);		
							sampleOutput.setVector(dimOut * offset, vecOut.getPointer(), dimOut);
							sampleWeight.setVector(offset, &weight, 1);
							sampleAlpha.setVector(offset, &alpha, 1);
#endif
							offset++;
						}
						//Finally update the number of samples if everything was successful
						numSamples += newNumSamples;
					}
					else
					{
						throw ios_base::failure("illegal numer of samples in file");
					}
				}
				else
				{
					throw ios_base::failure("dimensions in file do not fit!");
				}
				file.close(); //not really neccessary as destructor calls this automatically
			}
			/**
			* Saves the whole sample data to a binary file.
			* If the file already exits it will be overwritten.
			* @param *fileName         - Path to file which should be created
			* @throw ios_base::failure - if opening or writing the file fails
			* @return                  - none
			*/
			void KNearestNeighbor::saveFile(const char *fileName)
			{
				ofstream file;
				try
				{
					file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
					file.open(fileName, ios::out | ios::binary);
					writeStream(&file);
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
			/**
			* Loads sample data from an binary file
			* @param *fileName - Path to file which should be opened
			* @param append    - true if samples should be appended, false if samples should be replaced
			* @return          - none
			*/
			void KNearestNeighbor::loadFile(const char *fileName, bool append)
			{
				ifstream file;
				file.exceptions(ios_base::failbit | ios_base::eofbit | ios_base::badbit);
				file.open(fileName, ios::in | ios::binary);
				readStream(&file, append);
				file.close();
			}
#pragma endregion
#pragma region "Protected methods of class KNearestNeighbor"
			/**
			* Saves all samples to a stream
			* @param *file             - stream instance
			* @throw ios_base::failure - if opening or writing the file fails
			* @return                  - "true" if successfully, "false" if failed
			*/
			void KNearestNeighbor::writeStream(ofstream *file)
			{
				int32_t newNumSamples = numSamples, newDimIn = dimIn, newDimOut = dimOut; //Use int32_t (architecture independant)
				char magic[4] = {'k','N','N', '1'};	
				//write header
				file->write(magic, 4);
				file->write((char*)&newDimIn, sizeof(int32_t));
				file->write((char*)&newDimOut, sizeof(int32_t));
				file->write((char*)&newNumSamples, sizeof(int32_t));
				if (numSamples > 0) //Needed becuase std::vector raises exception when count is 0
				{
					//write data
					file->write((char*)&sampleInput[0],  sizeof(float) * dimIn  * numSamples);
					file->write((char*)&sampleOutput[0], sizeof(float) * dimOut * numSamples);
					file->write((char*)&sampleWeight[0], sizeof(float) * numSamples);
					file->write((char*)&sampleAlpha[0],  sizeof(float) * numSamples);
				}
			}
			/**
			* Reads samples from an binary stream
			* @param *file     - stream instance
			* @param append    - true if samples should be appended, false if samples should be replaced
			* @return          - "true" if successfully, "false" if failed
			*/
			void KNearestNeighbor::readStream(ifstream *file, bool append)
			{
				int32_t newNumSamples, newDimIn, newDimOut; //Use int32_t (architecture independant)
				char magic[4];

				file->read(magic, 4);
				file->read((char*)&newDimIn,      sizeof(int32_t));
				file->read((char*)&newDimOut,     sizeof(int32_t));
				file->read((char*)&newNumSamples, sizeof(int32_t));

				//magic must fit
				if ((magic[0] == 'k') && (magic[1] == 'N') && (magic[2] == 'N') && (magic[3] == '1'))
				{
					//there must be sample vectors in the file and the dimension must fit
					if ((newNumSamples > 0) && (newDimIn == dimIn) && (newDimOut == dimOut))
					{
						floatVector input (dimIn  * newNumSamples); // array to store all sample vectors temporary
						floatVector output(dimOut * newNumSamples); // array to store all output values temporary
						floatVector weight(newNumSamples); // array to store all output weight temporary
						floatVector alpha (newNumSamples); // array to store all output alpha temporary

						file->read((char*)&input[0],  sizeof(float) * newDimIn  * newNumSamples);
						file->read((char*)&output[0], sizeof(float) * newDimOut * newNumSamples);
						file->read((char*)&weight[0], sizeof(float) * newNumSamples);
						file->read((char*)&alpha[0],  sizeof(float) * newNumSamples);

						if (append)
						{
							//Append data...
							// append new data (offset is numSamples here)
#ifdef _USE_STD_VECTOR_CLASS
							this->sampleInput.resize ((numSamples + newNumSamples) * dimIn);
							this->sampleOutput.resize((numSamples + newNumSamples) * dimOut);
							this->sampleWeight.resize( numSamples + newNumSamples);
							this->sampleAlpha.resize ( numSamples + newNumSamples);
							memcpy(&sampleInput[numSamples],  &input[0],  dimIn  * newNumSamples * sizeof(float));
							memcpy(&sampleOutput[numSamples], &output[0], dimOut * newNumSamples * sizeof(float));
							memcpy(&sampleWeight[numSamples], &weight[0], newNumSamples * sizeof(float));
							memcpy(&sampleAlpha[numSamples],  &alpha[0],  newNumSamples * sizeof(float));
#else
							this->sampleInput.setVector (dimIn  * numSamples, &input[0],  dimIn  * newNumSamples);
							this->sampleOutput.setVector(dimOut * numSamples, &output[0], dimOut * newNumSamples);
							this->sampleWeight.setVector(numSamples,          &weight[0], newNumSamples);
							this->sampleAlpha.setVector (numSamples,          &alpha[0],  newNumSamples);
#endif
							//Set new sample count ONLY IF EVERYTHING worked correctly
							numSamples += newNumSamples; // increase numer of samples by the number of new samples
						}
						else
						{
							//No append...
							//Just replace the old pointers with the new ones (makes a copy internally)
							this->sampleInput  = input;
							this->sampleOutput = output;
							this->sampleWeight = weight;
							this->sampleAlpha  = alpha;
							numSamples = newNumSamples; // set new number of samples
						}
					}
					else
					{
						throw ios_base::failure("invalid file header (kNN1 expected)");
					}
				}
				else
				{
					throw ios_base::failure("header signature of file is unknown");
				}
			}
			/**
			* Calculate the squared euclidean distance between the two vectors (vectors are considered as points here)
			* @param *vecA - pointer to first vector
			* @param *vecB - pointer to second vector
			* @param dim   - dimension of both vectors
			* @return      - quadratic distance as float
			*/
			inline float KNearestNeighbor::vecDistQ(const float *vecA, const float *vecB, int dim)
			{
				float dist = 0.0f;
				for (int i = 0; i < dim; i++)
				{
					float dif = (*vecA++) - (*vecB++);
					dist += dif * dif;
				}
				return dist;
			}
			/**
			* kNN is an implentation of the k-Nearest Neighbor algorithm.
			* It searches the k (declared in numNeighbors) nearest neighbor vectors (in the declared samples array) for the input vector (*input).
			* The declared dimension must fit the dimension within the sample vectors in the sample array.
			* As metric the euclidean distance is used. The nearest neighbors are stored in ascending oder (entry with smallest distance first)
			* @param *input     - pointer to the input vector
			* @param dimension  - dimension of input vector and sample vectors
			* @param *samples   - pointer to sample vectors
			* @param numSamples - number of sample vectors available
			* @param *neighbors - pointer to an array of neighbors
			* @return           - none
			*/
			void KNearestNeighbor::kNN(const float *input, int dimension, const float *samples, int numSamples, Neighbors *neighbors)
			{
				float worstDist = FLT_MAX; //start with the worst possible distance 3.4*e^38
				//Neighbors must be initalized before!
				for (int i = 0; i < numSamples; i++)
				{
					float newDist = vecDistQ(input, samples, dimension);
					if (newDist < worstDist) // speed optimization (first check if the distance is smaller than the greatest distance in the neighbor array)
					{
						neighbors->insertNeighbor(i, newDist);
						worstDist = neighbors->getWorstDistance(); // distance of the last entry
					}
					samples += dimension; // move to next sample vector
				}
			}
			/**
			* Calculates the final output for the nearest neighbours determined by the kNN function.
			* The output is weighted by the quadratic distance to the input vector.
			* @param *sampleOutput - Pointer to an array of output values (index must fit to the index in sample vectors)
			* @param dimOutput     - dimension of final output vector
			* @param idxOutput     - current component index of the output vector
			* @param *sampleWeight - Pointer to an array of weight values (index must fit to the index in sample vectors)
			* @param *sampleAlpha  - Pointer to an array of "alpha" values (index must fit to the index in sample vectors)
			* @param numSamples    - number of sample vectors available
			* @param *neighbors    - pointer to an array of neighbors filled by a previous call to kNN()
			* @return output       - float value
			*/
			float KNearestNeighbor::kNNOutput(const float *sampleOutput, int dimOutput, int idxOutput, const float *sampleWeight, const float *sampleAlpha, int numSamples, Neighbors *neighbors)
			{
				Neighbors::NEIGHBOR* neighbor = neighbors->pointer;
				float result = 0.0f, weights = 0.0f;
				float weight;
				int idx;
				for (int i = 0; i < neighbors->numNeighbors; i++)
				{
					idx = neighbor->idx; // Index of the sample vector
					if (idx >= 0 && idx < numSamples) // Sanity check
					{
#ifdef _USE_GAUSSIAN_DISTRIBUTION					
						weight = 1.0f / (M_SQRT_PI * M_SQRT_2 * sampleAlpha[idx] ) * expf(- neighbor->distQ / (2.0f * sampleAlpha[idx] * sampleAlpha[idx]));
#else
						weight = sampleWeight[idx] * 1.0f / ( 1.0f + sampleAlpha[idx] * neighbor->distQ); //weight by distance^2
#endif					
						weights += weight;
						result += sampleOutput[idxOutput + idx * dimOutput] * weight; 
					}
					neighbor++; //next neighbor
				}
				if (weights != 0.0f)
				{
					result /= (weights); //divide by sum of all weights
				}
				else
				{
					return 0.0f;
				}
				return result;
			}
#pragma endregion
#pragma region "Protected methods of class Neighbors"
			/**
			* Adds the declared entry(index and distance) to the sorted array of NEIGHBORs
			* However, it will be added only if the distance is at least smaller than the greatest distance in the array.
			* @param idx          - index of the sample vector which should be added to the neighbor array
			* @param dist         - distance of the sample vector to the input vector 
			* @return             - none 
			*/
			inline void KNearestNeighbor::Neighbors::insertNeighbor(int idx, float dist)
			{
				NEIGHBOR* neighbors = this->pointer;
				for (int i = 0; i < numNeighbors; i++)
				{
					if (dist < neighbors->distQ)
					{
						//All lower entries must be moved down by one position (the lowest will be removed)
						memmove((void*)(neighbors + 1), (void*)neighbors, (numNeighbors - (i + 1)) * sizeof(NEIGHBOR)); //to be sure use memmove instead of memcpy because src and dest overlap (should not be neccessary here)
						//insert new entry
						neighbors->idx = idx;
						neighbors->distQ = dist;
						break; //make sure that at maximum one entry is replaced only
					}
					neighbors++;
				}
			}
			/**
			* Initializes all entries in the NEIGHBOR array with an invalid index(-1) and the maximum distance FLT_MAX
			* @param numNeighbors - number of elements the array should hold
			* @return             - none
			*/
			inline KNearestNeighbor::Neighbors::Neighbors(int numNeighbors)
			{
				NEIGHBOR* neighbors;
				this->numNeighbors = numNeighbors;
				this->pointer = new NEIGHBOR[numNeighbors];
				neighbors = this->pointer;
				for (int i = 0; i < numNeighbors; i++)
				{
					neighbors->idx = -1; // invalid index
					neighbors->distQ = FLT_MAX; //worst possible distance 3.4*e^38
					neighbors++; // next neighbor
				}
			}
			/**
			* Frees all reserved memory
			*/
			inline KNearestNeighbor::Neighbors::~Neighbors()
			{
				if (pointer)
				{
					delete [] pointer;
					pointer = NULL;
				}
				numNeighbors = 0;
			}
			/**
			* Returns the biggest distance in the NEIGHBORs array (squared)
			* @return - float value
			*/
			inline float KNearestNeighbor::Neighbors::getWorstDistance()
			{
				return pointer[numNeighbors - 1].distQ;
			}
#pragma endregion
		}
	}
}
