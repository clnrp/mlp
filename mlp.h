/*
 * mlp.h
 *
 *  Created on: 20/01/2011
 *      Author: cleoner
 */

#ifndef MLP_H_
#define MLP_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define sqr(x)        ((x)*(x))

class TLayer{
	public:
	int        Units;     // número de neurônios por camada
	double*    Output;    // sinal sináptico de cada neurônio
	double*    Error;     // erro de cada neurônio
	double**   Weight;    // peso de cada neurônio
	double**   dWeight;   // peso usado para calculo dos deltas, e para momentum
};

class TNet{
	public:
	//-----------
	TNet();
	~TNet();

	//-----------
	int      Layers;
	int*     Units;
	double   Error;
	double   Bias;
	double   Gain;
	double   Learning;
	double   Momentum;

	//-----------
	void     GenerateNetwork();
	void     SetInput(int Neuron,double Input);
	void     SetOutput(int Neuron,double Input);
	void     SetWeight(int Layer,int Neuron,int pWeight,double Weight);
	double   GetWeight(int Layer,int Neuron,int pWeight);
	void     RandomWeights();
	void     TrainNet();
	double*  TestNet();

	private:
	TLayer** Layer;
	double*  Target;
	double   GetOutput(int Neuron);
	void     PropagateNet();
	void     ComputeOutputError();
	void     BackpropagateNet();
	void     AdjustWeights();
};

#endif /* MLP_H_ */
