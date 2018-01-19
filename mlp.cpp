#include "mlp.h"

TNet::TNet(){
}

TNet::~TNet(){
	delete Target;
	for(int i=0; i<Layers; i++){
		delete Layer[i];
	}
}

void TNet::GenerateNetwork(){
	Target = new double[Units[Layers-1]];
	Layer = (TLayer**) calloc(Layers, sizeof(TLayer*));
	for(int i=0; i<Layers; i++){
		Layer[i] = (TLayer*) malloc(sizeof(TLayer));
		Layer[i]->Units      = Units[i];
		Layer[i]->Output     = (double*)  calloc(Units[i], sizeof(double));
		Layer[i]->Error      = (double*)  calloc(Units[i], sizeof(double));
		if (i > 0){
			Layer[i]->Weight     = (double**) calloc(Units[i], sizeof(double*));
			Layer[i]->dWeight    = (double**) calloc(Units[i], sizeof(double*));
			for(int j=0; j<Units[i]; j++){
				Layer[i]->Weight[j]  = (double*)  calloc(Units[i-1], sizeof(double));
				Layer[i]->dWeight[j] = (double*)  calloc(Units[i-1], sizeof(double));
			}
		}
	}
}

void TNet::SetInput(int Neuron, double Input){
	Layer[0]->Output[Neuron-1]=Input; // valor de um neurônio de entrada
}

void TNet::SetOutput(int Neuron, double Input){
	Target[Neuron-1]=Input; // valor de um neurônio de saída
}

double TNet::GetOutput(int Neuron){
	return Layer[Layers-1]->Output[Neuron-1]; // valor de um neurônio de saída
}

void TNet::SetWeight(int nLayer, int nNeuron, int pWeight, double Weight){
	Layer[nLayer-1]->Weight[nNeuron-1][pWeight-1]=Weight;
}

double TNet::GetWeight(int nLayer, int nNeuron, int pWeight){
	return Layer[nLayer-1]->Weight[nNeuron-1][pWeight-1];
}

void TNet::RandomWeights(){
	srand(100);
	for(int i=1; i<Layers; i++){ // layers a ser atualizados
		for(int j=0; j<Layer[i]->Units; j++){ // número de neurônios
			for(int l=0; l<Layer[i-1]->Units; l++){ // número de pesos por neurônio
				Layer[i]->Weight[j][l] = ((rand()%10)*pow(-1,floor(11*rand())))/10;
			}
		}
	}
}

void TNet::PropagateNet(){
	double Sum;
	for(int i=1; i<Layers; i++){ // layers a ser atualizados
		for(int j=0; j<Layer[i]->Units; j++){ // neurônios a ser sensibilizados
			Sum=0;
			for(int l=0; l<Layer[i-1]->Units; l++){ // número de pesos por neurônios, que é igual o número de neurônios do layer anterior
				Sum += Layer[i]->Weight[j][l] * Layer[i-1]->Output[l]; // Soma do produto entre sinais e pesos do neurônio
			}                                                      // Obs.: A soma não pode ter um valor elevado, pois se isso ocorrer,
																// teremos uma indeterminação na função de ativação
			Layer[i]->Output[j]=(1 / (1 + exp(-(Gain * Sum + Bias))));  // função de ativação
		}
	}
}

void TNet::ComputeOutputError(){ // Sinal do erro da camada de saída
	Error = 0;
	double Out=0,Err=0;
	for(int i=0; i<Layer[Layers-1]->Units; i++){ // número de neurônios da camada de saída
		Out = Layer[Layers-1]->Output[i];
		Err = Target[i]-Out; // esperado, menos o obtido
		Layer[Layers-1]->Error[i] = Gain * Out * (1-Out) * Err;
		Error += 0.5 * sqr(Err);
	}
}

void TNet::BackpropagateNet(){ // Sinal do erro das camadas intermediarias
	double Out,Err;
	for(int i=1; i<=(Layers-2); i++){ // número de layers intermediários
		for(int j=0; j<Layer[Layers-1-i]->Units; j++){ // número de pesos da camada superior
			Out = Layer[Layers-1-i]->Output[j]; // sinal do neurônio, da camada inferior
			Err = 0;
			for(int l=0; l<Layer[Layers-i]->Units; l++){ // número de neurônios da camada superior
			// Pesos de cada neurônio da camada superior, vezes o erro de cada neurônio dela
				Err = Err + Layer[Layers-i]->Weight[l][j] * Layer[Layers-i]->Error[l];
			}
			Layer[Layers-1-i]->Error[j] = Gain * Out * (1-Out) * Err;
		}
	}
}

void TNet::AdjustWeights(){
	double Out,Err,dWeight;
	for (int i=1; i<=(Layers-1); i++) { // layers a ser atualizados
		for (int j=0; j<Layer[i]->Units; j++) { // neurônios a ser sensibilizados
			for (int l=0; l<Layer[i-1]->Units; l++) { // indica o número de pesos por neurônio
				Out = Layer[i-1]->Output[l];  // valores de entrada
				Err = Layer[i]->Error[j];
				dWeight = Layer[i]->dWeight[j][l];
				Layer[i]->Weight[j][l] += Learning * Err * Out + Momentum * dWeight;
				Layer[i]->dWeight[j][l] = Learning * Err * Out;
			}
		}
	}
}

void TNet::TrainNet(){
	PropagateNet(); // propaga o sinal de entrada ate a saída

	ComputeOutputError(); // erro da camada de saída
	BackpropagateNet(); // propaga erro para camadas intermediarias
	AdjustWeights(); // ajusta os pesos
}

double* TNet::TestNet(){
	double* Output;
	Output = new double[Layer[Layers-1]->Units];

	PropagateNet(); // propaga o sinal de entrada ate a saída
	for(int i=0; i<Layer[Layers-1]->Units; i++){
		Output[i] = GetOutput(i+1);
	}
	return Output;
}
