#include "math-binary.h"


//swd~~~~~~~~~~~~random number from -1 to 1
double getRand(){
    return (( (double)rand() * 2 ) / ( (double)RAND_MAX + 1 ) ) - 1;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//swd~~~~~~~~~~~~get the deterministic binary function
int getBinary(double B_input) {
    int B_output;
    if(B_input>=0){
        B_output=1;
    }
    else{
        B_output=-1;
    }
    return B_output;
}

//swd~~~~~~~~~~~~get the clip function
double getClip(double C_input) {
    double C_output;
    if(C_input>=1){
        C_output=1;
    }
    else if(C_input<=-1){
        C_output=-1;
    }
    else {
        C_output=C_output;
    }
    return C_output;
}

//swd~~~~~~~~~~~~calculating the Rsquared value of the regression ML result
float Rsquared(double y_label[], double f_actual[], int Rsize){
    int i_rquare;

    double value1=0, value2=0, value3=0;
    for(i_rquare=0; i_rquare<Rsize; i_rquare++){
        value1=value1+ y_label[i_rquare]*f_actual[i_rquare];
        value2=value2+ y_label[i_rquare]*y_label[i_rquare];
        value3=value3+ f_actual[i_rquare]*f_actual[i_rquare];
    }
 
    double Rsquared_value=0;


    Rsquared_value=(value1*value1)/(value2*value3);
    printf("so the Rsquared value is : %f \n", Rsquared_value);
    return Rsquared_value;
}
