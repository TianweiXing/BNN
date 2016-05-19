#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include <conio.h>

#include "BinaryMLP.h"
#include "math-binary.h"

#define TRAINING_ITERATION 5000000


int main() {

    double inputs[MAX_NO_OF_INPUTS];
    double outputTargets[MAX_NO_OF_OUTPUTS];

    /* determine layer paramaters */
    int noOfLayers = 2; // input layer excluded
    int noOfNeurons[] = {10,1};
    int noOfInputs[] = {2,10};
    char axonFamilies[] = {'g','l'};
    double actFuncFlatnesses[] = {1,1,1};

    createNet(noOfLayers, noOfNeurons, noOfInputs, axonFamilies, actFuncFlatnesses, 1);

    /* train it using batch method */
    int i;
    double tempTotal1, tempTotal2;
    int counter = 0;
    for(i = 0; i < TRAINING_ITERATION; i++) {
        inputs[0] = getRand();
        inputs[1] = getRand();
        inputs[2] = getRand();
        inputs[3] = getRand();
        tempTotal1 = inputs[0] + inputs[1];
        tempTotal2 = inputs[2] - inputs[3];
        // tempTotal = inputs[0] + inputs[1];
        feedNetInputs(inputs);
        updateNetOutput();
        // outputTargets[0] = sin(tempTotal1)*2+0.5*exp(tempTotal2)-cos(inputs[1]+inputs[3])/2;
        outputTargets[0] = sin(tempTotal1)*2+0.5*exp(tempTotal2);
 //       outputTargets[1] = (double)cos(tempTotal);
        /* train using batch training ( don't update weights, just cumulate them ) */
        //trainNet(0, 0, 1, outputTargets);
        trainNet(0, 0, 1, outputTargets);
        counter++;
        /* apply batch changes after 100 loops use .8 learning rate and .8 momentum */
        if(counter == 100) { applyBatchCumulations(.3,.3); counter = 0;}  //!~~~swd: should be within (0,1)
    }

    /* test it */
    double *outputs;
    double target_out[50];
    double actual_out[50];
    printf("Sin Target \t Output \n");
    printf("---------- \t -------- \t ---------- \t --------\n");
    for(i = 0; i < 50; i++) {
        inputs[0] = getRand();
        inputs[1] = getRand();
        inputs[2] = getRand();
        inputs[3] = getRand();
        tempTotal1 = inputs[0] + inputs[1];
        tempTotal2 = inputs[2] - inputs[3];

	target_out[i] = sin(tempTotal1)*2+0.5*exp(tempTotal2);
    // target_out[i] = sin(tempTotal1)*2+0.5*exp(tempTotal2)-cos(inputs[1]+inputs[3])/2;

        feedNetInputs(inputs);
        updateNetOutput();
        outputs = getOutputs();

	actual_out[i] = outputs[0];
	

        printf( "%f \t %f \n", target_out[i], actual_out[i]);
    }
    
    float Rsquared_ans=Rsquared(target_out, actual_out, 50);
    printf("Result Summary: \n");
    printf("The Rsquared Value is : %f \n", Rsquared_ans);
    printf("finish!!!\n");

    //getch();
    return 0;

}
