#ifndef MATH_BINARY_H
#define MATH_BINARY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double getRand();

int getBinary(double B_input);

double getClip(double C_input);

float Rsquared(double y_label[], double f_actual[], int Rsize);

#endif
