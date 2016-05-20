#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define KERNEL_SIZE 3
// convolution kernel structure
struct conv_kernel{
    int kernel_dim;
    double kernel_para[KERNEL_SIZE][KERNEL_SIZE];   
};

// image patch structure
struct img_patch{
	double value[KERNEL_SIZE][KERNEL_SIZE];
};

// // Function for convolutional operation
// double ConvolutionFunc(double input_img_slice, struct conv_kernel input_kernel, int img_dim1, int img_dim2){

//     int kernel_size;
//     int img_new_dim1;
//     int img_new_dim2;
//     kernel_size=input_kernel.kernel_dim;
//     img_new_dim1=img_dim1-(kernel_size - 1);
//     img_new_dim2=img_dim2-(kernel_size - 1);

//     // a new result image for storing convolution result
//     double result_img[img_new_dim1][img_new_dim2];

//     for (int i=0; i<img_new_dim1; i++){
//         for (int j=0; j<img_new_dim2; j++){

//             // do multiplication-addition for a single window
//             double addMulti=0;
//             for (int m=0; m<kernel_size; m++){
//                 for (int n=0; n<kernel_size; n++){
//                     addMulti = input_kernel.kernel_para[m][n]* result_img[i+m][j+n] + addMulti;
//                 }
//             }

//             result_img[i][j]=addMulti;
//         }
//     }

//     // return result_img[][];
//     return 0;
// }

double ConvolutionFunc(struct conv_kernel input_kernel, struct img_patch image_patch){
	int kernel_size;
	kernel_size=input_kernel.kernel_dim;
    // do multiplication-addition for a single window
    double convolution_result=0;
    for (int m=0; m<kernel_size; m++){
        for (int n=0; n<kernel_size; n++){
            convolution_result = input_kernel.kernel_para[m][n]* image_patch.value[m][n] + convolution_result;
        }
    }


	return convolution_result;
}



int main(){
	int a=1;
	int b=0;

	if ((a==1)&(b==1)){
		printf("dashabi\n");
	}
	else{
		printf("yidinggaowan\n");
	}

	int image_example[4][4];
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			image_example[i][j]=4*i+j;
			printf("%d\t",image_example[i][j]);
		}
		printf("\n");
	}

	printf("\n");

	struct conv_kernel kernel_example;
	kernel_example.kernel_dim=3;
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			kernel_example.kernel_para[i][j]=3*i+j;
			printf("%1.f\t",kernel_example.kernel_para[i][j]);
		}
		printf("\n");
	}



    printf("\n");
    printf("\n");


	double result_image[2][2];

	for(int i=0; i<2; i++){
		for(int j=0; j<2; j++){
			int kernel_size = kernel_example.kernel_dim;
			struct img_patch image_patch;

			// set each image_patch
			for(int m=0; m<3; m++){
				for(int n=0; n<3; n++){
					image_patch.value[m][n]= image_example[i+m][j+n];
				}
			}

			// calculate the convolution for this window
			result_image[i][j]= ConvolutionFunc(kernel_example, image_patch);

			printf("%.f\t",result_image[i][j]);
		}
		printf("\n");
	}
	



	
	return 0;
}