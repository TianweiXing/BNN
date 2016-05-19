#ifndef BINARYMLP_H
#define BINARYMLP_H

#include "math-binary.h"

#define MAX_NO_OF_LAYERS 6
#define MAX_NO_OF_INPUTS 6
#define MAX_NO_OF_NEURONS 100
#define MAX_NO_OF_WEIGHTS 1000
#define MAX_NO_OF_OUTPUTS 1

struct neuron{
    double *output;
    double threshold;
    double oldThreshold;
    double batchCumulThresholdChange;
    char axonFamily;
    double *weights;
    double *oldWeights;
    double *netBatchCumulWeightChanges;
    int noOfInputs;
    double *inputs;
    double actFuncFlatness;
    double *error;
};

struct layer {
    int noOfNeurons;
    struct neuron * neurons;
};

static struct neuralNet {
    int noOfInputs;
    double *inputs;
    double *outputs;
    int noOfLayers;
    struct layer *layers;
    int noOfBatchChanges;
} theNet;


//func declearation
void createNet( int, int *, int *, char *, double *, int );
void feedNetInputs(double *);
void updateNetOutput(void);
double *getOutputs();
void trainNet ( double, double, int, double * );
void applyBatchCumulations( double, double );
int loadNet(char *);
int saveNet(char *);

static struct neuron netNeurons[MAX_NO_OF_NEURONS];
static double netInputs[MAX_NO_OF_INPUTS];
static double netNeuronOutputs[MAX_NO_OF_NEURONS];
static double netErrors[MAX_NO_OF_NEURONS];
static struct layer netLayers[MAX_NO_OF_LAYERS];
static double netWeights[MAX_NO_OF_WEIGHTS];
static double netOldWeights[MAX_NO_OF_WEIGHTS];
static double netBatchCumulWeightChanges[MAX_NO_OF_WEIGHTS];

#endif
