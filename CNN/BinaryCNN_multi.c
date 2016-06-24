// gcc BinaryCNN.c -o a.out -lm -std=c99


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#pragma GCC diagnostic ignored "-Wwrite-strings"

// function for weight binarization
int binaryFun(float x){
    if (x>=0){
        return 1;
    }
    else{
        return -1;
    }
}

// code for activation binarization function
float min_value(float x, float y){
    if(x<=y){
        return x;
    }
    else{
        return y;
    }
}
float max_value(float x, float y){
    if(x<=y){
        return y;
    }
    else{
        return x;
    }
}
float binary_tanh_unit(float x){
    float result;

    float temp= (x+1)/2;
    temp = min_value(temp, 1);
    temp = max_value(temp, 0);
    result= 2* round(temp) -1;

    return result;
}

// 
#define EPSILON 1e-4

// define input images numbers
#define IMAGE_NUM 4

// definition of image input
#define IMAGE_ROW 32
#define IMAGE_COL 32
#define IMAGE_DIMENSION 3
//  image_dimension repressent the 3 dimensional RGB value

// define array length for ordinary array use
#define ARRAY_LENGTH 10

// definition of convolution kernels
#define KERNEL_SIZE 3
#define KERNEL_MAX_DEPTH 512
//  image_dimension repressent the 3 dimensional RGB value

// image structure
struct cifar_img{
    float pixel_img[IMAGE_DIMENSION][IMAGE_ROW][IMAGE_COL];
    int label_img;
};

// convolution kernel structure
struct conv_kernel{
    int kernel_dim;
    int kernel_depth;
    float kernel_para[KERNEL_MAX_DEPTH][KERNEL_SIZE][KERNEL_SIZE];
    
};

// image patch structure
struct img_patch{
    int patch_depth;
    float value[KERNEL_MAX_DEPTH][KERNEL_SIZE][KERNEL_SIZE];
};

// generate random number: either -1 to 1
float getRand(){
    // binary setting:
    // srand((unsigned)time(NULL));
    // return (round( (float)rand()/RAND_MAX )*2-1);
    // real number setting:
    return (( (float)rand() * 2 ) / ( (float)RAND_MAX + 1 ) ) - 1;
}

static float ReadWeightVector[1024];  // created for saving bias/ mean/ var/ gamma/ beta, not for weights larger than 1024
// read parameters (bias, gamma, beta, mean, inv_std) from binary files
int readWeights(char file_name[], int weight_length){
    FILE *fp = fopen(file_name, "rb");

    fread(&ReadWeightVector,sizeof(float)*weight_length, 1, fp);

    fclose(fp);

    // printf("%.5f\n",ReadWeightVector[0]);

    return 0;
}


static float FCWeightVector[8196];  // created for saving fully-connected weight, not for weights larger than 8196
// read FC-Layer weights parameters from binary files
int readFCweights(char file_name[], int weight_length, int weight_position, int jump_num){
    FILE *fp = fopen(file_name, "rb");
    float temp_c;

    fseek(fp,4*(weight_position),SEEK_CUR);

    for (int i=0; i<weight_length; i++){
        fread(&temp_c,sizeof(temp_c), 1, fp);
        FCWeightVector[i] = binaryFun(temp_c);

        fseek(fp,4*jump_num,SEEK_CUR);
    }

    fclose(fp);
    return 0;
}



// use a static struct for fully connected layers
#define MAX_NEURON_NUM 10000

static struct NNLayer_list {
    int valid_list_index;
    float list_values[MAX_NEURON_NUM];
}NNLayer;


// function for load kernel parameters 
// use random generated initial parameters: set ifRand=1 (by default)
struct conv_kernel LoadKernel(int ker_size, int ker_depth, int kernel_position, char kernel_flie[]){
    // in python array, the kernel values are stored in the opposite way (for col and row)

    // to set the kernel values from txt parameter files 
    // exported from python CNN model
    int kernel_tot_length=ker_depth*ker_size*ker_size;
    float kernel_data[kernel_tot_length]; 

    FILE *fp = fopen(kernel_flie, "rb");
    fseek(fp,4*kernel_tot_length*(kernel_position),SEEK_SET);

    fread(&kernel_data,sizeof(kernel_data), 1, fp);
    fclose(fp);


    struct conv_kernel set_kernel;
    set_kernel.kernel_dim = ker_size;
    set_kernel.kernel_depth=ker_depth;

    int h, i, j;

    // h is the kernel depth
    for (h = 0; h<set_kernel.kernel_depth; h++){
    
        // i is the kernel row            
        for(i=0; i<set_kernel.kernel_dim; i++){
            // j is the kernel col
            for(j=0; j<set_kernel.kernel_dim; j++){
                    set_kernel.kernel_para[h][set_kernel.kernel_dim-i-1][set_kernel.kernel_dim-j-1]=binaryFun(kernel_data[ h*9 + i*3 + j ]);
                    // printf("%f\t",set_kernel.kernel_para[h][i][j]);
            }
        }
    }
    return set_kernel;
}




// use a static struct instead
#define MAX_DIMENSION_1 512
#define MAX_DIMENSION_2 32
#define MAX_DIMENSION_3 32

static struct image_map {
    int valid_dim_1;
    int valid_dim_2;
    int valid_dim_3;
    float mapping_values[MAX_DIMENSION_1][MAX_DIMENSION_2][MAX_DIMENSION_3];
}ImageMap;

static struct image_map temp_ImageMap;

// initialization for ImageMap in the Main function


// Function for convolutional operation
double ConvolutionFunc(struct conv_kernel input_kernel, float conv_bias, struct img_patch image_patch){
    int kernel_size, kernel_dep;
    kernel_size=input_kernel.kernel_dim;
    kernel_dep=input_kernel.kernel_depth;

    if (kernel_dep!=image_patch.patch_depth){
        printf("kernel_dep is: %d. \t image patch depth is: %d\n",kernel_dep,image_patch.patch_depth);
        printf("inconsistent kernel size and image patch!\n");
    }

    // do multiplication-addition for a single window
    float convolution_result=0;

    int l,m,n;
    for (l=0; l<kernel_dep; l++){

        for ( m=0; m<kernel_size; m++){
            for ( n=0; n<kernel_size; n++){
                convolution_result = input_kernel.kernel_para[l][m][n]* image_patch.value[l][m][n] + convolution_result;
            }
        }
    }

    convolution_result = convolution_result+conv_bias;

    return convolution_result;
}


// function for convolution layer
// unable to test it yet (question about paper)
void Conv2DLayer(int kernel_size, int num_kernel, int pad, char kernel_flie[], char bias_file[]){
    int actual_dim1=ImageMap.valid_dim_1;
    int actual_dim2=ImageMap.valid_dim_2 + 2 * pad;
    int actual_dim3=ImageMap.valid_dim_3 + 2 * pad;

    // update ImageMap valid dimensions
    // all kernels in the same layer has same size
    ImageMap.valid_dim_1= num_kernel;
    ImageMap.valid_dim_2= actual_dim2 -(kernel_size-1);
    ImageMap.valid_dim_3= actual_dim3 -(kernel_size-1);


    // if(binary){
    //     printf("using binary kernel to do convolution");
    //     // change the kernels to binary kernels
    // }

    float bias[num_kernel];

    readWeights( bias_file,  num_kernel);
    for (int i=0; i<num_kernel; i++){
        bias[i]=ReadWeightVector[i];
    }

    // do convolution for every single kernel
    for(int i=0; i<num_kernel; i++){

        // cannot store an array of this size in memory
        // so we read the parameters of a single kernel every time
        struct conv_kernel input_kernel;
        float input_conv_bias;
        // input_kernel.kernel_dim = kernel_size;
        // input_kernel.kernel_depth = actual_dim1;
        input_kernel = LoadKernel(kernel_size, actual_dim1, i, kernel_flie);

// ~~~~~~~~~~~~~~~~~~~~~~~~for checking
        // printf("the %d th kernel :\n", i); 
        // for(int ii=0; ii<3; ii++){
        // for(int jj=0; jj<input_kernel.kernel_dim; jj++){
        //     for(int kk=0; kk<input_kernel.kernel_dim; kk++){
        //         printf("%1.5f\t",input_kernel.kernel_para[ii][jj][kk]);
        //     }
        //     printf("\n");
        // }
        // printf("\n\n");
        // }  
        
// ~~~~~~~~~~~~~~~~~~~~~~~~for checking


        // load the convolution bias from file;
        input_conv_bias = bias[i];



        float image_slice[actual_dim1][actual_dim2][actual_dim3];

        // initialize image slice using image_map stored value
        for (int l=0; l<actual_dim1; l++){

            for(int m=0; m<actual_dim2; m++){
                for (int n=0; n<actual_dim3; n++){
                    // if on the pad edge, set the value to 0
                    if ((m==0)|(n==0)|(m==(actual_dim2-1))|(n==((actual_dim3-1)))) {
                        image_slice[l][m][n] = 0;
                    }
                    else{
                        image_slice[l][m][n] = ImageMap.mapping_values[l][m-1][n-1];
                    }

                }
            }

        }

        // create result image slice 
        // currently we are not considering stride here: default stride is 1;
        int result_dim1 = num_kernel ;
        int result_dim2 = actual_dim2 -(kernel_size-1);
        int result_dim3 = actual_dim3 -(kernel_size-1);
        float result_image[result_dim2][result_dim3];

        for(int k=0; k<result_dim2; k++){
            for(int l=0; l<result_dim3; l++){
                
                struct img_patch image_patch;

                image_patch.patch_depth=actual_dim1;

                // set each image_patch (3D)
                for(int m=0; m<kernel_size; m++){
                    for(int n=0; n<kernel_size; n++){
                        for (int o=0; o<actual_dim1; o++){
                            image_patch.value[o][m][n]= image_slice[o][k+m][l+n];
                        }
                    }
                }

                // calculate the convolution for this window
                result_image[k][l]= ConvolutionFunc(input_kernel, input_conv_bias, image_patch);
                // printf("%1.3f\t",result_image[k][l]);
            }
            // printf("\n");
        }
        

        // // update the mapping values of ImageMap
        // for (int m=0; m<ImageMap.valid_dim_2; m++){
        //     for (int n=0; n<ImageMap.valid_dim_3; n++){
        //         ImageMap.mapping_values[i ][m][n]=result_image[m][n];
        //     }
        // }
        // // for every actual dimension


        // store the feature mapping values of ImageMap to temp_ImageMap
        for (int m=0; m<ImageMap.valid_dim_2; m++){
            for (int n=0; n<ImageMap.valid_dim_3; n++){
                temp_ImageMap.mapping_values[i ][m][n]=result_image[m][n];
            }
        }
        // for every actual dimension

    
    }
    // for every kernel

    // update the mapping values of ImageMap
    for (int l=0; l<ImageMap.valid_dim_1; l++){
        for (int m=0; m<ImageMap.valid_dim_2; m++){
            for (int n=0; n<ImageMap.valid_dim_3; n++){
                ImageMap.mapping_values[l][m][n]=temp_ImageMap.mapping_values[l][m][n];
            }
        }
    }
}

// function for finding the maximun number in an 1D array
float MaxNum(float array[], int length){
    float max_number=array[0];
    for (int i=0; i<length; i++){
        if(array[i]>max_number){
            max_number=array[i];
        }
    }
    return max_number;
}


// function for Max pooling layer
// by default, the stride value is 0, pad is (0,0)
void MaxPooling(int pool_size){

    int new_dim1=ImageMap.valid_dim_1;
    int new_dim2=ImageMap.valid_dim_2 / pool_size;
    int new_dim3=ImageMap.valid_dim_3 / pool_size;

    // for every image slice in ImageMap, use maxpooling to reduce the size of image
    for(int i=0; i<new_dim1; i++){
        float pooled_image[new_dim2][new_dim2];
        float pooled_array[ARRAY_LENGTH];

        for(int j=0; j<new_dim2; j++){
            for(int k=0; k<new_dim3; k++){

                // get all the value in the pooling window to a pooling array
                for (int m=0; m<pool_size; m++){
                    for (int n=0; n<pool_size; n++){
                        pooled_array[m*pool_size+n]=ImageMap.mapping_values[i][pool_size*j+m][pool_size*k+n];

                        // printf("%.f ",pooled_array[m*pool_size+n]);
                    }
                }
                
                // do max pooling and get new pixel number
                pooled_image[j][k]=MaxNum(pooled_array, pool_size * pool_size);

                // printf("\t %.f %d %d ",pooled_image[j][k], j, k);
                // printf("\n");

                // update image_map mapping values
                ImageMap.mapping_values[i][j][k]=pooled_image[j][k];
            }
        }
    }

    // parameter update for ImageMap
    ImageMap.valid_dim_2=new_dim2;
    ImageMap.valid_dim_3=new_dim3;

}


// to implemet activation for every entry after convolution
void ActivationLayer(char layer_type,char activation){

    switch (layer_type){

        case 'c':
            for(int dim=0; dim<ImageMap.valid_dim_1; dim++)
            {
                for(int i=0; i<ImageMap.valid_dim_2; i++)
                {
                    for(int j=0; j<ImageMap.valid_dim_3; j++)
                    {   
                        float temp;
                        temp = ImageMap.mapping_values[dim][i][j];
                        // for different activation
                        // use binary activation in binary mode
                        switch (activation){
                            case 't': 
                            // hyper tangent function 

                                    ImageMap.mapping_values[dim][i][j] = binary_tanh_unit(temp);
                                
                                break;
                            default:
                                break;
                        }

                    }
                }
            }
            break; 

        case 'd':

            for(int j=0; j<NNLayer.valid_list_index; j++)
            {   
                float temp;
                temp = NNLayer.list_values[j];
                // for different activation
                // use binary activation in binary mode
                switch (activation){
                    case 't': 
                    // hyper tangent function 
                            NNLayer.list_values[j] = binary_tanh_unit(temp);
                        
                        break;
                    default:
                        break;
                }

            }
            break;

        default:
            break;

    }    
}



// function for batchnormalization layer
// currently only consider forward propogation
// consider batch size is 1
void BatchNormLayer(int para_num, char Normalayer, char beta_flie[], char gamma_flie[], char mean_flie[], char invstd_flie[]){

	int i, j ,k;
	float x, y;

	// LoadBatchNormPara
	// from saved files
	float beta[para_num];
    float gamma[para_num];
    float mean[para_num];
    float inv_std[para_num];

    // read beta parameters
	readWeights( beta_flie,  para_num);
    for (i=0; i<para_num; i++){
		beta[i]=ReadWeightVector[i];
	}
    // read gamma parameters
    readWeights( gamma_flie,  para_num);
    for (i=0; i<para_num; i++){
        gamma[i]=ReadWeightVector[i];
    }
    // read mean parameters
    readWeights( mean_flie,  para_num);
    for (i=0; i<para_num; i++){
        mean[i]=ReadWeightVector[i];
    }
    // read invstd parameters
    readWeights( invstd_flie,  para_num);
    for (i=0; i<para_num; i++){
        inv_std[i]=ReadWeightVector[i];
    }



    if(Normalayer == 'C'){
        int dim1=ImageMap.valid_dim_1;
        int dim2=ImageMap.valid_dim_2;
        int dim3=ImageMap.valid_dim_3;

        for (i=0; i<dim1; i++){
            for (j=0; j<dim2; j++){
                for (k=0; k<dim3; k++){
                    x=ImageMap.mapping_values[i][j][k];
                    y = (x - mean[i]) * inv_std[i] * gamma[i] + beta[i];
                    ImageMap.mapping_values[i][j][k] = y;
                }
            }
        }
    }
    else if (Normalayer == 'F'){
        int dim = NNLayer.valid_list_index;

        for (i = 0; i<dim; i++){
            x=NNLayer.list_values[i];
            y = (x - mean[i]) * inv_std[i] * gamma[i] + beta[i];
            NNLayer.list_values[i] = y;
        }

    }
    else{
        printf("Invalid Normalization!\n");
    }


// not for Lasagne use: 
	// if(Normalayer == 'C'){
	// 	// For CNN, different elements of the same feature map at different
	// 	// locations,  are  normalized in  the  same  way;
	// 	// we  jointly  normalize  all  the  activations  in  a  mini-batch, over all locations
	// 	// We learn a pair of parameters gamma & beta per feature map, rather than per activation
	// 	// During inference the BN transform applies the same linear
	// 	// transformation to each activation in a given feature map.

	// 	int dim1=ImageMap.valid_dim_1;
	// 	int dim2=ImageMap.valid_dim_2;
	// 	int dim3=ImageMap.valid_dim_3;

	//     for (i=0; i<dim1; i++){

	//     	// compute the mean of current feature map
	//     	int sum=0;
	//     	for (j=0; j<dim2; j++){
	//             for (k=0; k<dim3; k++){
	//             	sum = sum+ImageMap.mapping_values[i][j][k];
	//             }
	//         }
	//         mean = sum /(dim2 * dim3);

	//         // compute the variance of current feature map
	//         sum=0;
	//     	for (j=0; j<dim2; j++){
	//             for (k=0; k<dim3; k++){
	//             	sum = sum + pow((ImageMap.mapping_values[i][j][k] - mean), 2);
	//             }
	//         }
	//         variance = variance /(dim2 * dim3);

	//         // implement batch normalization (also linear transform)
	//         for (j=0; j<dim2; j++){
	//             for (k=0; k<dim3; k++){
	//             	x=ImageMap.mapping_values[i][j][k];

	//             	y = (x - mean)/(sqrt( variance + EPSILON )) * gamma[i] + beta[i];
	//             	ImageMap.mapping_values[i][j][k] = y;
	//             }
	//         }
	//     }
	// }
	// else if (Normalayer == 'F'){
	// 	int dim = NNLayer.valid_list_index;
	// 	for (i = 0; i<dim; i++){
	// 		x=NNLayer.list_values[i];

	// 		// for dense layer, if batch size is 1, then mean = x, variance =0
	// 		// implement batch normalization (also linear transform)
	// 	    if (batch_size == 1){
 //        		mean = x;
 //        		variance = 1;
 //        	}

 //        	y = (x - mean)/(sqrt( variance + EPSILON )) * gamma[i] + beta[i];
 //        	NNLayer.list_values[i] = y;
	// 	}

	// }
	// else{
	// 	printf("Invalid Normalization!\n");
	// }
    
}


// function for resize of 3D array to 1D array
// transfer the valid values from ImageMap to NNLayer
void ResizeMapping2List(){

    // the dimension change
    int dim1=ImageMap.valid_dim_1;
    int dim2=ImageMap.valid_dim_2;
    int dim3=ImageMap.valid_dim_3;
    int list_dimension=dim1*dim2*dim3;

    NNLayer.valid_list_index = list_dimension;

    int i, j, k;
    int index = 0;
    for (i=0; i<dim1; i++){
        for (j=0; j<dim2; j++){
            for (k=0; k<dim3; k++){

                NNLayer.list_values[index] = ImageMap.mapping_values[i][j][k];
                index = index +1;
            }
        }
    }

    if (index != list_dimension){
        printf("ResizeMapping2List went wrong!\n");
    }

}


// // function for load denseLayer weights parameters 
// // use random generated initial parameters: set ifRand=1 (by default)
// double LoadWeight(double weight_list[], int input_num, int output_num, int init_weight=1 ){
//     // to set the weight values from txt parameter files 
//     // exported from python CNN model


//     int i, j;
//     if(init_kernel){

//         // set the random values for every kernel
//         set_kernel=input_kernel;
//         // i is the kernel row            
//         for(i=0; i<set_kernel.kernel_dim; i++){
//             // j is the kernel col
//             for(j=0; j<set_kernel.kernel_dim; j++){
//                     set_kernel.kernel_para[i][j]=getRand();
//             }
//         }

//     }
//     return set_kernel;
// }


// function for creating a fully connected layer(dense layer)
// change / update the denseLayer after add-mutiplication

// void DenseLayer(int output_node_num,  int binary=0, int init_weight=1){
void DenseLayer(int output_node_num, char weight_file[], char bias_file[]){

    int input_node_num;
    input_node_num = NNLayer.valid_list_index;
    // cannot store an array of this size in memory
    // float weight_array[output_node_num][input_node_num];
    // load weight parameters from file  
    // if (init_weight){
    //     for (i=0; i<output_node_num; i++){
    //         for (j=0; j<input_node_num; j++){

    //             weight_array[i][j] = getRand();
    //         }
    //     }
    // }
    // creating a new list to store the output
    float output_values[output_node_num];
    float bias[output_node_num];

    int i, j;

    readWeights( bias_file,  output_node_num);
    for (i=0; i<output_node_num; i++){
        bias[i]=ReadWeightVector[i];
    }
    // calculating the weighted sum value
    for (i=0; i<output_node_num; i++){

        // for every output neuron, load the weight parameters from file
        float weight_array[input_node_num];
        float neuron_bias;

        readFCweights(weight_file, input_node_num, i, (output_node_num-1));
        for (j = 0; j<input_node_num; j++){
            weight_array[j] = FCWeightVector[j];
        }


        neuron_bias = bias[i];


        float weighted_sum =0;
        for (j=0; j<input_node_num; j++){

            weighted_sum = weighted_sum + NNLayer.list_values[j] * weight_array[j];
        }

        output_values[i] = weighted_sum + neuron_bias;

        // ~~~~~~~~~~~~~~~for checking
        // printf("the %d  th output value is %.4f\n",i,output_values[i]);

    }

    // store output_values to NNLayer
    NNLayer.valid_list_index = output_node_num;

    for (i=0; i<output_node_num; i++){

        NNLayer.list_values[i] = output_values[i];
    }
}



int main(){

    int test_num = 3;
    int img_index;
    float accuracy=0;
    int correct_count=0;
    for (img_index=0; img_index<test_num; img_index++){


    int img_position = img_index+1;
   // read cifar data from binary file

    unsigned char buffer[3073];
    FILE *ptr;

    ptr = fopen("test_batch.bin","rb");  // r for read, b for binary

    fseek(ptr, sizeof(buffer)*(img_position-1),SEEK_CUR); // go to the position of specific image

    fread(buffer,sizeof(buffer),1,ptr); // read 10 bytes to our buffer


    for(int i = 0; i<3073; i++)
    {
        // printf("%u ", buffer[i]); // prints a series of bytes
    }
    fclose(ptr);

    // save cifar data to image structure
    struct cifar_img img_eg;
    img_eg.label_img=buffer[0];
    // printf("%u\n",buffer[0]);
    printf("the label of example image is :%d\n",img_eg.label_img);
    
    unsigned char zero='0';
    int temp_pixel;
    float std_pixel;

    for(int dim=0; dim<IMAGE_DIMENSION; dim++)
    {
        for(int i=0; i<IMAGE_ROW; i++)
        {
            for(int j=0; j<IMAGE_COL; j++)
            {
                // save values from binary to image struct
                temp_pixel=buffer[ dim*1024 + 32*i +j + 1]-zero + 48;
                // when input image data, change the scale from 0-255 to -1 - +1;
                std_pixel = (temp_pixel*2.0/255.0) - 1.0;
                img_eg.pixel_img[dim][i][j]=std_pixel;
                // print image to the terminal
                // printf("%.5f\t",img_eg.pixel_img[dim][i][j]);
            }
            // printf("\n");
        }
        // printf("\n\n");
    }

    

    // // kernel function test
    // // the max_kernel_size is set to 3, so here the dimension should be smaller than 3
    // struct conv_kernel kernel_eg[2];
    // kernel_eg[0].kernel_dim=3;
    // kernel_eg[1].kernel_dim=2;

    // // LoadKernel(int ker_size, int ker_depth, int init_kernel=1 )

    // for(int i=0; i<2; i++){
    //     kernel_eg[i]=LoadKernel(3,1, 1);

    //     for(int j=0; j<kernel_eg[i].kernel_dim; j++){
    //         for(int k=0; k<kernel_eg[i].kernel_dim; k++){
    //             printf("%1.3f\t",kernel_eg[i].kernel_para[0][j][k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");
    // }
    
//     // #######################################################################################
//     // #######################################################################################
//     // #############################  Start of Main program  #################################
//     // #######################################################################################  
//     // #######################################################################################


// # input is 32*32 RGB image

    // for start,the image_map is just the cifar10-image
    for (int i=0; i<IMAGE_DIMENSION; i++){
        for (int j=0; j<IMAGE_ROW; j++){
            for(int k=0; k<IMAGE_COL; k++){
                ImageMap.mapping_values[i][j][k]=img_eg.pixel_img[i][j][k];
            }
        }
    }

    ImageMap.valid_dim_1=IMAGE_DIMENSION;
    ImageMap.valid_dim_2=IMAGE_ROW;
    ImageMap.valid_dim_3=IMAGE_COL;

    // printf("Input of cifar-10 image, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");




    // # 128C3-128C3-P2    
    printf("\n");     
    // example of creating kernels, loading kernels, convolution, maxpooling and Nonlinear activation;
    // struct conv_kernel conv_128_1[128];
    // void Conv2DLayer(int kernel_size, int num_kernel, int pad, int binary=0)
    // Conv2DLayer(3, 128, 1);
    Conv2DLayer(3, 128, 1,  "arr_0", "arr_1");
    // printf("2D Convolutional layer, with 128 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    

    // BatchNormLayer(1, 128, 'C');
    BatchNormLayer(128, 'C', "arr_2", "arr_3", "arr_4", "arr_5");
    // BatchNormLayer(128, 'C', "arr_0", "arr_0", "arr_0", "arr_0");
    ActivationLayer('c', 't');



    Conv2DLayer(3, 128, 1,  "arr_6", "arr_7");
    // printf("2D Convolutional layer, with 128 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    



    MaxPooling(2);
    // printf("Max-pooling layer, with 2*2 pooling size, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    BatchNormLayer( 128, 'C', "arr_8", "arr_9", "arr_10", "arr_11");




    ActivationLayer('c', 't');


    // # 256C3-256C3-P2   
    printf("\n"); 

    Conv2DLayer(3, 256, 1,  "arr_12", "arr_13");
    // printf("2D Convolutional layer, with 256 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    BatchNormLayer(256, 'C', "arr_14", "arr_15", "arr_16", "arr_17");

    ActivationLayer('c', 't');


    Conv2DLayer(3, 256, 1,  "arr_18", "arr_19");
    // printf("2D Convolutional layer, with 256 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    MaxPooling(2);
    // printf("Max-pooling layer, with 2*2 pooling size, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    BatchNormLayer( 256, 'C', "arr_20", "arr_21", "arr_22", "arr_23");

    ActivationLayer('c', 't');


    // # 512C3-512C3-P2  
    // printf("\n");    

    Conv2DLayer(3, 512, 1,  "arr_24", "arr_25");
    // printf("2D Convolutional layer, with 512 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    BatchNormLayer(512, 'C', "arr_26", "arr_27", "arr_28", "arr_29");

    ActivationLayer('c', 't');


    Conv2DLayer(3, 512, 1,  "arr_30", "arr_31");
    // printf("2D Convolutional layer, with 512 3*3 kernels, the size of ImageMap is :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    MaxPooling(2);
    // printf("Max-pooling layer, with 2*2 pooling size, the size of ImageMap is  :\n");
    // printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    // printf("\n");

    BatchNormLayer( 512, 'C', "arr_32", "arr_33", "arr_34", "arr_35");

    ActivationLayer('c', 't');


    


    ResizeMapping2List();
    // printf("Resize the ImageMap to 1D array NNlayer, the size of NeuralNet-Layer is :\n");
    // printf("%d",NNLayer.valid_list_index);
    // printf("\n");


    // # 1024FP-1024FP-10FP     
    // printf("\n");
    DenseLayer(1024, "arr_36", "arr_37");
    // DenseLayer(1024);
    // printf("Fully Connected Layer with 1024 neurons, the size of NeuralNet-Layer is :\n");
    // printf("%d",NNLayer.valid_list_index);
    // printf("\n");

    // for(int k=0; k<NNLayer.valid_list_index; k++){
    //     printf("%.4f\t", NNLayer.list_values[k]);
    // }
    // printf("\n");


    BatchNormLayer( 1024, 'F', "arr_38", "arr_39", "arr_40", "arr_41");

    ActivationLayer('d', 't');


    DenseLayer(1024, "arr_42", "arr_43");
    // printf("Fully Connected Layer with 1024 neurons, the size of NeuralNet-Layer is :\n");
    // printf("%d",NNLayer.valid_list_index);
    // printf("\n");

    BatchNormLayer(1024, 'F', "arr_44", "arr_45", "arr_46", "arr_47");

    ActivationLayer('d','t');


    DenseLayer(10, "arr_48", "arr_49");
    // printf("Fully Connected Layer with 10 neurons, the size of NeuralNet-Layer is :\n");
    // printf("%d",NNLayer.valid_list_index);
    // printf("\n");
    BatchNormLayer(10, 'F', "arr_50", "arr_51", "arr_52", "arr_53");


    // printf("\n\n\n");
    int max_ind=1;
    float max_value=NNLayer.list_values[0];
    for (int i=0; i<10; i++){
        if(NNLayer.list_values[i] >max_value ){
            max_ind = i;
            max_value =NNLayer.list_values[i];
        }
        // printf("%.5f\t", NNLayer.list_values[i]);
    }
    printf("\nthe estimated label is : %d \n", max_ind);
    printf("the label of example image is :%d\n",img_eg.label_img);

    printf("count +1: %d\n",correct_count);
    if (max_ind == img_eg.label_img){
        correct_count++;
        
    }
       
    accuracy = correct_count/(img_index+1.0);
    printf("accuracy: %.3f\n", accuracy);

    }
    return 0;
}


// Input of cifar-10 image, the size of ImageMap is :
// 3   32  32  

// 2D Convolutional layer, with 128 3*3 kernels, the size of ImageMap is :
// 128 32  32  
// 2D Convolutional layer, with 128 3*3 kernels, the size of ImageMap is :
// 128 32  32  
// Max-pooling layer, with 2*2 pooling size, the size of ImageMap is :
// 128 16  16  

// 2D Convolutional layer, with 256 3*3 kernels, the size of ImageMap is :
// 256 16  16  
// 2D Convolutional layer, with 256 3*3 kernels, the size of ImageMap is :
// 256 16  16  
// Max-pooling layer, with 2*2 pooling size, the size of ImageMap is :
// 256 8   8   

// 2D Convolutional layer, with 512 3*3 kernels, the size of ImageMap is :
// 512 8   8   
// 2D Convolutional layer, with 512 3*3 kernels, the size of ImageMap is :
// 512 8   8   
// Max-pooling layer, with 2*2 pooling size, the size of ImageMap is  :
// 512 4   4   
// Resize the ImageMap to 1D array NNlayer, the size of NeuralNet-Layer is :
// 8192

// Fully Connected Layer with 1024 neurons, the size of NeuralNet-Layer is :
// 1024
// Fully Connected Layer with 1024 neurons, the size of NeuralNet-Layer is :
// 1024
// Fully Connected Layer with 10 neurons, the size of NeuralNet-Layer is :
// 10
// [Finished in 6.6s]