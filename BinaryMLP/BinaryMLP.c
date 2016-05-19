#include "BinaryMLP.h"


void createNet( int noOfLayers, int *noOfNeurons, int *noOfInputs, char *axonFamilies, double *actFuncFlatnesses, int initWeights ) {

    int i, j, counter, counter2, counter3, counter4;
    int totalNoOfNeurons, totalNoOfWeights;

    theNet.layers = netLayers;
    theNet.noOfLayers = noOfLayers;
    theNet.noOfInputs = noOfInputs[0];
    theNet.inputs = netInputs;

    //swd ~~~~initialization of neurons, neuron output, and weight
    totalNoOfNeurons = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {
        totalNoOfNeurons += noOfNeurons[i];
    }
    for(i = 0; i < totalNoOfNeurons; i++) { netNeuronOutputs[i] = 0; }

    totalNoOfWeights = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {
        totalNoOfWeights += noOfInputs[i] * noOfNeurons[i];
    }

    counter = counter2 = counter3 = counter4 = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {
        for(j = 0; j < noOfNeurons[i]; j++) {
            if(i == theNet.noOfLayers-1 && j == 0) { // beginning of the output layer
                theNet.outputs = &netNeuronOutputs[counter]; //swd ~~~~return the actual adress of netNeuronOutputs[counter]
            }
            netNeurons[counter].output = &netNeuronOutputs[counter];
            netNeurons[counter].noOfInputs = noOfInputs[i];
            netNeurons[counter].weights = &netWeights[counter2];
            netNeurons[counter].netBatchCumulWeightChanges = &netBatchCumulWeightChanges[counter2];
            netNeurons[counter].oldWeights = &netOldWeights[counter2];
            netNeurons[counter].axonFamily = axonFamilies[i];
            netNeurons[counter].actFuncFlatness = actFuncFlatnesses[i];
            if ( i == 0) {
                netNeurons[counter].inputs = netInputs; //swd~~~~ this is the 1st hidden layer
            }
            else {
                netNeurons[counter].inputs = &netNeuronOutputs[counter3];
            }
            netNeurons[counter].error = &netErrors[counter];
            counter2 += noOfInputs[i];
            counter++;
        }
        netLayers[i].noOfNeurons = noOfNeurons[i];
        netLayers[i].neurons = &netNeurons[counter4];
        if(i > 0) {
            counter3 += noOfNeurons[i-1];
        }
        counter4 += noOfNeurons[i];
    }

    // initialize weights and thresholds
     if ( initWeights == 1 ) {
        for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].threshold = getRand(); }
        for( i = 0; i < totalNoOfWeights; i++) { netWeights[i] = getRand(); }
        for( i = 0; i < totalNoOfWeights; i++) { netOldWeights[i] = netWeights[i]; }
        for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].oldThreshold = netNeurons[i].threshold; }
    }

    // initialize batch values
    for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].batchCumulThresholdChange = 0; }
    for( i = 0; i < totalNoOfWeights; i++) { netBatchCumulWeightChanges[i] = 0; }
    theNet.noOfBatchChanges = 0;

}

//swd ~~~~feed the input values to theNet
void feedNetInputs(double *inputs) {
     int i;
     for ( i = 0; i < theNet.noOfInputs; i++ ) {
        netInputs[i] = inputs[i];
     }
}

//swd~~~get the output of a single neuron
static void updateNeuronOutput(struct neuron * myNeuron, int Current_layer, int noOfLayers) {

    double activation = 0;
    int i;

    for ( i = 0; i < myNeuron->noOfInputs; i++) {
        activation += myNeuron->inputs[i] * getBinary(myNeuron->weights[i]);//swd~~~~let the weight be binary value
        // activation += myNeuron->inputs[i] * (myNeuron->weights[i]);//swd~~~~let the weight be double value
    }
    activation += -1 * myNeuron->threshold;//swd ~~~~what is activation: Ax+b;  the threshold (b) not be binary

    double temp;   
    switch (myNeuron->axonFamily) {
        case 'g': // logistic
            temp = -activation / myNeuron->actFuncFlatness;
            /* avoid overflow */
            if ( temp > 45 ) {
                *(myNeuron->output) = 0;
            }
            else if ( temp < -45 ) {
                *(myNeuron->output) = 1;
            }
            else {
                *(myNeuron->output) = 1.0 / ( 1 + exp( temp ));//swd~~~~this is the neuron output place
            }
            // if(Current_layer != (noOfLayers-1) ){
            //     *(myNeuron->output) = getBinary(*(myNeuron->output));//swd~~~~if it's not the output layer, let the neuron output be binary
            // }

            break;
        case 't': // tanh
            temp = -activation / myNeuron->actFuncFlatness;
            /* avoid overflow */
            if ( temp > 45 ) {
                *(myNeuron->output) = -1;
            }
            else if ( temp < -45 ) {
                *(myNeuron->output) = 1;
            }
            else {
                *(myNeuron->output) = ( 2.0 / ( 1 + exp( temp ) ) ) - 1;
            }
            // if(Current_layer!=noOfLayers-1){
            //     *(myNeuron->output) = getBinary(*(myNeuron->output));//swd~~~~if it's not the output layer, let the neuron output be binary
            // }
            break;
        case 'l': // linear
            *(myNeuron->output) = activation;
            // if(Current_layer!=noOfLayers-1){
            //     *(myNeuron->output) = getBinary(*(myNeuron->output));//swd~~~~if it's not the output layer, let the neuron output be binary
            // }
            break;
        default:
            break;
    }

}

//swd~~~~update every neuron output in theNet
void updateNetOutput( ) {

    int i, j;

    for(i = 0; i < theNet.noOfLayers; i++) {
        for( j = 0; j < theNet.layers[i].noOfNeurons; j++) {
            updateNeuronOutput(&(theNet.layers[i].neurons[j]),i, theNet.noOfLayers);
        }
    }

}

// swd~~~~get the derivative of the the 3 types of standard Activation Funcs
static double derivative (struct neuron * myNeuron) {

    double temp;
    switch (myNeuron->axonFamily) {
        case 'g': // logistic
            temp = ( *(myNeuron->output) * ( 1.0 - *(myNeuron->output) ) ) / myNeuron->actFuncFlatness; break;
        case 't': // tanh
            temp = ( 1 - pow( *(myNeuron->output) , 2 ) ) / ( 2.0 * myNeuron->actFuncFlatness ); break;
        case 'l': // linear
            temp = 1; break;
        default:
            temp = 0; break;
    }
    return temp;

}

// learningRate and momentumRate will have no effect if batch mode is 'on'
void trainNet ( double learningRate, double momentumRate, int batch, double *outputTargets ) {

    int i,j,k;
    double temp;
    struct layer *currLayer, *nextLayer;

     // calculate errors
    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        if ( i == theNet.noOfLayers - 1 ) { // output layer
            for ( j = 0; j < currLayer->noOfNeurons; j++ ) {
                *(currLayer->neurons[j].error) = derivative(&currLayer->neurons[j]) * ( outputTargets[j] - *(currLayer->neurons[j].output));
            }
        }
        else { // other layers
            nextLayer = &theNet.layers[i+1];
            for ( j = 0; j < currLayer->noOfNeurons; j++ ) {
                temp = 0;
                for ( k = 0; k < nextLayer->noOfNeurons; k++ ) {
                    temp += *(nextLayer->neurons[k].error) * nextLayer->neurons[k].weights[j];
                }
                *(currLayer->neurons[j].error) = derivative(&currLayer->neurons[j]) * temp;
            }
        }
    }

    // update weights n thresholds
    double tempWeight;
    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        for ( j = 0; j < currLayer->noOfNeurons; j++ ) {

            // thresholds
            if ( batch == 1 ) {
                    currLayer->neurons[j].batchCumulThresholdChange += *(currLayer->neurons[j].error) * -1;
            }
            else {
                tempWeight = currLayer->neurons[j].threshold;
                currLayer->neurons[j].threshold += ( learningRate * *(currLayer->neurons[j].error) * -1 ) + ( momentumRate * ( currLayer->neurons[j].threshold - currLayer->neurons[j].oldThreshold ) );
                currLayer->neurons[j].oldThreshold = tempWeight;
            }

            // weights
            if ( batch == 1 ) {
                for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                    currLayer->neurons[j].netBatchCumulWeightChanges[k] +=  *(currLayer->neurons[j].error) * currLayer->neurons[j].inputs[k];
                }
            }
            else {
                for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                    tempWeight = currLayer->neurons[j].weights[k];
                    currLayer->neurons[j].weights[k] += ( learningRate * *(currLayer->neurons[j].error) * currLayer->neurons[j].inputs[k] ) + ( momentumRate * ( currLayer->neurons[j].weights[k] - currLayer->neurons[j].oldWeights[k] ) );
                    currLayer->neurons[j].weights[k] = getClip(currLayer->neurons[j].weights[k]);
                    currLayer->neurons[j].oldWeights[k] = tempWeight;
                }

            }

        }
    }

    if(batch == 1) {
        theNet.noOfBatchChanges++;
    }

}

void applyBatchCumulations( double learningRate, double momentumRate ) {

    int i,j,k;
    struct layer *currLayer;
    double tempWeight;

    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        for ( j = 0; j < currLayer->noOfNeurons; j++ ) {
            // thresholds
            tempWeight = currLayer->neurons[j].threshold;
            currLayer->neurons[j].threshold += ( learningRate * ( currLayer->neurons[j].batchCumulThresholdChange / theNet.noOfBatchChanges ) ) + ( momentumRate * ( currLayer->neurons[j].threshold - currLayer->neurons[j].oldThreshold ) );
            currLayer->neurons[j].oldThreshold = tempWeight;
            currLayer->neurons[j].batchCumulThresholdChange = 0;
            // weights
            for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                tempWeight = currLayer->neurons[j].weights[k];
                currLayer->neurons[j].weights[k] += ( learningRate * ( currLayer->neurons[j].netBatchCumulWeightChanges[k] / theNet.noOfBatchChanges ) ) + ( momentumRate * ( currLayer->neurons[j].weights[k] - currLayer->neurons[j].oldWeights[k] ) );
                // currLayer->neurons[j].weights[k] = getClip(currLayer->neurons[j].weights[k]);
                currLayer->neurons[j].oldWeights[k] = tempWeight;
                currLayer->neurons[j].netBatchCumulWeightChanges[k] = 0;
            }
        }
    }

    theNet.noOfBatchChanges = 0;

}

//swd ~~~get the output of theNet
double *getOutputs() {

    return theNet.outputs;

}

//swd~~~~this func never been used????
int loadNet(char *path) {

    int tempInt; double tempDouble; char tempChar;
    int i, j, k;

    int noOfLayers;
    int noOfNeurons[MAX_NO_OF_LAYERS];
    int noOfInputs[MAX_NO_OF_LAYERS];
    char axonFamilies[MAX_NO_OF_LAYERS];
    double actFuncFlatnesses[MAX_NO_OF_LAYERS];

    FILE *inFile;

    if(!(inFile = fopen(path, "rb")))
    return 1;

    fread(&tempInt,sizeof(int),1,inFile);
    noOfLayers = tempInt;

    for(i = 0; i < noOfLayers; i++) {

        fread(&tempInt,sizeof(int),1,inFile);
        noOfNeurons[i] = tempInt;

        fread(&tempInt,sizeof(int),1,inFile);
        noOfInputs[i] = tempInt;

        fread(&tempChar,sizeof(char),1,inFile);
        axonFamilies[i] = tempChar;

        fread(&tempDouble,sizeof(double),1,inFile);
        actFuncFlatnesses[i] = tempDouble;

    }

    createNet(noOfLayers, noOfNeurons, noOfInputs, axonFamilies, actFuncFlatnesses, 0);

    // now the weights
    for(i = 0; i < noOfLayers; i++) {
        for (j = 0; j < noOfNeurons[i]; j++) {
            fread(&tempDouble,sizeof(double),1,inFile);
            theNet.layers[i].neurons[j].threshold = tempDouble;
            for ( k = 0; k < noOfInputs[i]; k++ ) {
                fread(&tempDouble,sizeof(double),1,inFile);
                theNet.layers[i].neurons[j].weights[k] = tempDouble;
            }
        }
    }

    fclose(inFile);

    return 0;

}
