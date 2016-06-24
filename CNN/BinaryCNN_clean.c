// we set all images/kernels value to float type
// in binary implementation, they should have only binary values.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#pragma GCC diagnostic ignored "-Wwrite-strings"

// #######################################################################################
// All relevant math operation
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

// function for finding the maximun number in an 1D array
float MaxNum(float array[], int length){
    float max_number=array[0];
    int i;
    for (i=0; i<length; i++){
        if(array[i]>max_number){
            max_number=array[i];
        }
    }
    return max_number;
}

// generate random number: either -1 to 1
float getRand(){
    return (( (float)rand() * 2 ) / ( (float)RAND_MAX + 1 ) ) - 1;
}


// #######################################################################################
// define input images numbers
#define IMAGE_NUM 1
// definition of image input size
// image_dimension repressent the 3 dimensional RGB value
#define IMAGE_ROW 32
#define IMAGE_COL 32
#define IMAGE_DIMENSION 3
// define array length for ordinary array use
#define ARRAY_LENGTH 10
// definition of convolution kernels size
#define KERNEL_SIZE 3
#define KERNEL_MAX_DEPTH 512
// definition of weight vector length
#define WEIGHT_MAX_IN_NUM 8192

// image structure: to store image input
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

struct fc_weight{
    int input_num;
    float weight_para[WEIGHT_MAX_IN_NUM];
};

// #######################################################################################
// define FC weight vector sets for 3 fully-connected layers
#define fc_weight_num_1 1024
struct fc_weight fullConectWeight_1[fc_weight_num_1];
#define fc_weight_num_2 1024
struct fc_weight fullConectWeight_2[fc_weight_num_2];
#define fc_weight_num_3 10
struct fc_weight fullConectWeight_3[fc_weight_num_3];

// define globle weight vector sets :(1+4)*9 vectors
#define general_para_num 45
struct fc_weight general_parameter[general_para_num];

// define kernel weights structs for 6 convolution layers
#define kernel_num_1 128
struct conv_kernel cnn_kernel_1[kernel_num_1];
#define kernel_num_2 128
struct conv_kernel cnn_kernel_2[kernel_num_2];
#define kernel_num_3 256
struct conv_kernel cnn_kernel_3[kernel_num_3];
#define kernel_num_4 256
struct conv_kernel cnn_kernel_4[kernel_num_4];
#define kernel_num_5 512
struct conv_kernel cnn_kernel_5[kernel_num_5];
#define kernel_num_6 512
struct conv_kernel cnn_kernel_6[kernel_num_6];


// #######################################################################################
// Functions for loading trained parameters:
// read FC-Layer weights parameters from binary files
struct fc_weight readFCweights(int input_num, int weight_position, int jump_num, char file_name[]){

    struct fc_weight set_weight;
    set_weight.input_num = input_num;

    FILE *fp = fopen(file_name, "rb");
    float temp_c;
    int i;

    fseek(fp,4*(weight_position),SEEK_SET);
    for (i=0; i<input_num; i++){
        fread(&temp_c,sizeof(temp_c), 1, fp);
        set_weight.weight_para[i] = binaryFun(temp_c);
        fseek(fp,4*(jump_num-1),SEEK_CUR);
    }
    fclose(fp);
    return set_weight;
}

// read general weights parameters from binary files:(bias, gamma, beta, mean, inv_std)
struct fc_weight readWeights(int weight_length, char file_name[]){
    struct fc_weight set_weight;
    set_weight.input_num = weight_length;

    FILE *fp = fopen(file_name, "rb");
    float temp_c;
    int i;
    for (i=0; i<weight_length; i++){
        fread(&temp_c,sizeof(temp_c), 1, fp);
        set_weight.weight_para[i] = temp_c;
    }
    fclose(fp);
    return set_weight;
}

// function for load kernel parameters 
struct conv_kernel LoadKernel(int ker_size, int ker_depth, int kernel_position, char kernel_flie[]){
    // in python array, the kernel values are stored in the opposite way (for col and row)
    // exported from python BinaryCNN model
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
            }
        }
    }
    return set_kernel;
}


// #######################################################################################
// use a static struct to store feature map, updated after every CNN layer
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

// use a static struct for fully connected layers, updated after every dense layer
#define MAX_NEURON_NUM 10000

static struct NNLayer_list {
    int valid_list_index;
    float list_values[MAX_NEURON_NUM];
}NNLayer;


// #######################################################################################
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
void Conv2DLayer(int kernel_size, int num_kernel, int pad, struct conv_kernel cnn_kernel[], struct fc_weight bias_struct){

    int actual_dim1=ImageMap.valid_dim_1;
    int actual_dim2=ImageMap.valid_dim_2 + 2 * pad;
    int actual_dim3=ImageMap.valid_dim_3 + 2 * pad;

    // update ImageMap valid dimensions
    // all kernels in the same layer has same size
    ImageMap.valid_dim_1= num_kernel;
    ImageMap.valid_dim_2= actual_dim2 -(kernel_size-1);
    ImageMap.valid_dim_3= actual_dim3 -(kernel_size-1);

    float bias[num_kernel];
    int j=0;
    for ( j=0; j<num_kernel; j++){
        bias[j]=bias_struct.weight_para[j];
    }

    // do convolution for every single kernel
    int i, k, l, m, n, o;
    for( i=0; i<num_kernel; i++){
        struct conv_kernel input_kernel;
        float input_conv_bias;
        // load the convolution kernel and bias;
        input_kernel = cnn_kernel[i];
        input_conv_bias = bias[i];

        float image_slice[actual_dim1][actual_dim2][actual_dim3];
        // initialize image slice using image_map stored value
        for ( l=0; l<actual_dim1; l++){
            for( m=0; m<actual_dim2; m++){
                for ( n=0; n<actual_dim3; n++){
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

        for( k=0; k<result_dim2; k++){
            for( l=0; l<result_dim3; l++){
                
                struct img_patch image_patch;
                image_patch.patch_depth=actual_dim1;
                // set each image_patch (3D)
                for( m=0; m<kernel_size; m++){
                    for( n=0; n<kernel_size; n++){
                        for ( o=0; o<actual_dim1; o++){
                            image_patch.value[o][m][n]= image_slice[o][k+m][l+n];
                        }
                    }
                }

                // calculate the convolution for this window
                result_image[k][l]= ConvolutionFunc(input_kernel, input_conv_bias, image_patch);
            }
        }
        
        // store the feature mapping values of ImageMap to temp_ImageMap
        for ( m=0; m<ImageMap.valid_dim_2; m++){
            for ( n=0; n<ImageMap.valid_dim_3; n++){
                temp_ImageMap.mapping_values[i ][m][n]=result_image[m][n];
            }
        }
        // for every actual dimension
    }
    // for every kernel

    // update the mapping values of ImageMap
    for ( l=0; l<ImageMap.valid_dim_1; l++){
        for ( m=0; m<ImageMap.valid_dim_2; m++){
            for ( n=0; n<ImageMap.valid_dim_3; n++){
                ImageMap.mapping_values[l][m][n]=temp_ImageMap.mapping_values[l][m][n];
            }
        }
    }
}


// function for Max pooling layer
// by default, the stride value is 0, pad is (0,0)
void MaxPooling(int pool_size){

    int new_dim1=ImageMap.valid_dim_1;
    int new_dim2=ImageMap.valid_dim_2 / pool_size;
    int new_dim3=ImageMap.valid_dim_3 / pool_size;

    int i, j, k, m, n;
    // for every image slice in ImageMap, use maxpooling to reduce the size of image
    for( i=0; i<new_dim1; i++){
        float pooled_image[new_dim2][new_dim2];
        float pooled_array[ARRAY_LENGTH];

        for( j=0; j<new_dim2; j++){
            for( k=0; k<new_dim3; k++){

                // get all the value in the pooling window to a pooling array
                for ( m=0; m<pool_size; m++){
                    for ( n=0; n<pool_size; n++){
                        pooled_array[m*pool_size+n]=ImageMap.mapping_values[i][pool_size*j+m][pool_size*k+n];
                    }
                }
                
                // do max pooling and get new pixel number
                pooled_image[j][k]=MaxNum(pooled_array, pool_size * pool_size);
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

    int i, j, dim;

    switch (layer_type){

        case 'c':
            for( dim=0; dim<ImageMap.valid_dim_1; dim++)
            {
                for( i=0; i<ImageMap.valid_dim_2; i++)
                {
                    for( j=0; j<ImageMap.valid_dim_3; j++)
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
            for( j=0; j<NNLayer.valid_list_index; j++)
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
// consider batch size is 1, load trained parameter, use the distribution of training data
void BatchNormLayer(int para_num, char Normalayer, struct fc_weight beta_struct,struct fc_weight gamma_struct,struct fc_weight mean_struct, struct fc_weight invstd_struct){

	int i, j ,k;
	float x, y;
	// LoadBatchNormPara from saved files
	float beta[para_num];
    float gamma[para_num];
    float mean[para_num];
    float inv_std[para_num];

    // read beta parameters
    for (i=0; i<para_num; i++){
		beta[i]=beta_struct.weight_para[i];
	}
    // read gamma parameters
    for (i=0; i<para_num; i++){
        gamma[i]=gamma_struct.weight_para[i];
    }
    // read mean parameters
    for (i=0; i<para_num; i++){
        mean[i]=mean_struct.weight_para[i];
    }
    // read invstd parameters
    for (i=0; i<para_num; i++){
        inv_std[i]=invstd_struct.weight_para[i];
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


// function for creating a fully connected layer(dense layer)
// change / update the denseLayer after add-mutiplication
void DenseLayer(int output_node_num,  struct fc_weight fc_weight_struct[], struct fc_weight bias_struct){

    int input_node_num;
    input_node_num = NNLayer.valid_list_index;

    // creating a new list to store the output
    float output_values[output_node_num];
    float bias[output_node_num];

    int i, j;

    for (i=0; i<output_node_num; i++){
        bias[i]=bias_struct.weight_para[i];
    }
    // calculating the weighted sum value
    for (i=0; i<output_node_num; i++){
        // for every output neuron, load the weight parameters from file
        float weight_array[input_node_num];
        float neuron_bias;
        for (j = 0; j<input_node_num; j++){
            weight_array[j] = fc_weight_struct[i].weight_para[j];
        }
        neuron_bias = bias[i];

        float weighted_sum =0;
        for (j=0; j<input_node_num; j++){

            weighted_sum = weighted_sum + NNLayer.list_values[j] * weight_array[j];
        }

        output_values[i] = weighted_sum + neuron_bias;
    }

    // store output_values to NNLayer
    NNLayer.valid_list_index = output_node_num;
    for (i=0; i<output_node_num; i++){
        NNLayer.list_values[i] = output_values[i];
    }
}



int main(){

    // set timer to estimate computation time
    time_t start, timer_0, timer_end;
    start = clock();

    // loading all trained parameters:
    // loading kernel parameters
    char kernel_file_1[]="arr_0";
    char kernel_file_2[]="arr_6";
    char kernel_file_3[]="arr_12";
    char kernel_file_4[]="arr_18";
    char kernel_file_5[]="arr_24";
    char kernel_file_6[]="arr_30";

    int i;
    for( int i=0; i <kernel_num_1; i++){
        cnn_kernel_1[i] = LoadKernel(3,3,i, kernel_file_1);
    }
    for( i=0; i <kernel_num_2; i++){
        cnn_kernel_2[i] = LoadKernel(3,128,i, kernel_file_2);
    }
    for( i=0; i <kernel_num_3; i++){
        cnn_kernel_3[i] = LoadKernel(3,128,i, kernel_file_3);
    }
    for( i=0; i <kernel_num_4; i++){
        cnn_kernel_4[i] = LoadKernel(3,256,i, kernel_file_4);
    }
    for( i=0; i <kernel_num_5; i++){
        cnn_kernel_5[i] = LoadKernel(3,256,i, kernel_file_5);
    }
    for( i=0; i <kernel_num_6; i++){
        cnn_kernel_6[i] = LoadKernel(3,512,i, kernel_file_6);
    }

    // loading dense layer weight parameters
    char fc_weight_file_1[]="arr_36";
    char fc_weight_file_2[]="arr_42";
    char fc_weight_file_3[]="arr_48";
    for( i=0; i <fc_weight_num_1; i++){
        fullConectWeight_1[i] = readFCweights(8196, i, 1024, fc_weight_file_1);
    }
    for( i=0; i <fc_weight_num_2; i++){
        fullConectWeight_2[i] = readFCweights(1024, i, 1024, fc_weight_file_2);
    }
    for( i=0; i <fc_weight_num_3; i++){
        fullConectWeight_3[i] = readFCweights(1024, i, 10, fc_weight_file_3);
    }

    // loading other general parameters
    // parameters are stored as this sequence: bias, beta, gamma, mean, inv_std. 5 groups of params for a layer
    char bias_file_1[]="arr_1";
    char beta_file_1[]="arr_2";
    char gamma_file_1[]="arr_3";
    char mean_file_1[]="arr_4";
    char invstd_file_1[]="arr_5";

    char bias_file_2[]="arr_7";
    char beta_file_2[]="arr_8";
    char gamma_file_2[]="arr_9";
    char mean_file_2[]="arr_10";
    char invstd_file_2[]="arr_11";

    char bias_file_3[]="arr_13";
    char beta_file_3[]="arr_14";
    char gamma_file_3[]="arr_15";
    char mean_file_3[]="arr_16";
    char invstd_file_3[]="arr_17";

    char bias_file_4[]="arr_19";
    char beta_file_4[]="arr_20";
    char gamma_file_4[]="arr_21";
    char mean_file_4[]="arr_22";
    char invstd_file_4[]="arr_23";

    char bias_file_5[]="arr_25";
    char beta_file_5[]="arr_26";
    char gamma_file_5[]="arr_27";
    char mean_file_5[]="arr_28";
    char invstd_file_5[]="arr_29";

    char bias_file_6[]="arr_31";
    char beta_file_6[]="arr_32";
    char gamma_file_6[]="arr_33";
    char mean_file_6[]="arr_34";
    char invstd_file_6[]="arr_35";

    char bias_file_7[]="arr_37";
    char beta_file_7[]="arr_38";
    char gamma_file_7[]="arr_39";
    char mean_file_7[]="arr_40";
    char invstd_file_7[]="arr_41";

    char bias_file_8[]="arr_43";
    char beta_file_8[]="arr_44";
    char gamma_file_8[]="arr_45";
    char mean_file_8[]="arr_46";
    char invstd_file_8[]="arr_47";

    char bias_file_9[]="arr_49";
    char beta_file_9[]="arr_50";
    char gamma_file_9[]="arr_51";
    char mean_file_9[]="arr_52";
    char invstd_file_9[]="arr_53";

    // int para_size_array[9]= {128, 128, 256, 256, 512, 512, 1024, 1024, 10};
    // int temp_size_ind;

    // for (int i=0; i< general_para_num; i++){
    //     temp_size_ind = i%9;
    //     general_parameter[i]=readWeights(para_size_array[temp_size_ind],  general_param_file[]);
    // }

    general_parameter[0]=readWeights(128,  bias_file_1);
    general_parameter[1]=readWeights(128,  beta_file_1);
    general_parameter[2]=readWeights(128,  gamma_file_1);
    general_parameter[3]=readWeights(128,  mean_file_1);
    general_parameter[4]=readWeights(128,  invstd_file_1);

    general_parameter[5]=readWeights(128,  bias_file_2);
    general_parameter[6]=readWeights(128,  beta_file_2);
    general_parameter[7]=readWeights(128,  gamma_file_2);
    general_parameter[8]=readWeights(128,  mean_file_2);
    general_parameter[9]=readWeights(128,  invstd_file_2);

    general_parameter[10]=readWeights(256,  bias_file_3);
    general_parameter[11]=readWeights(256,  beta_file_3);
    general_parameter[12]=readWeights(256,  gamma_file_3);
    general_parameter[13]=readWeights(256,  mean_file_3);
    general_parameter[14]=readWeights(256,  invstd_file_3);

    general_parameter[15]=readWeights(256,  bias_file_4);
    general_parameter[16]=readWeights(256,  beta_file_4);
    general_parameter[17]=readWeights(256,  gamma_file_4);
    general_parameter[18]=readWeights(256,  mean_file_4);
    general_parameter[19]=readWeights(256,  invstd_file_4);

    general_parameter[20]=readWeights(512,  bias_file_5);
    general_parameter[21]=readWeights(512,  beta_file_5);
    general_parameter[22]=readWeights(512,  gamma_file_5);
    general_parameter[23]=readWeights(512,  mean_file_5);
    general_parameter[24]=readWeights(512,  invstd_file_5);

    general_parameter[25]=readWeights(512,  bias_file_6);
    general_parameter[26]=readWeights(512,  beta_file_6);
    general_parameter[27]=readWeights(512,  gamma_file_6);
    general_parameter[28]=readWeights(512,  mean_file_6);
    general_parameter[29]=readWeights(512,  invstd_file_6);

    general_parameter[30]=readWeights(1024,  bias_file_7);
    general_parameter[31]=readWeights(1024,  beta_file_7);
    general_parameter[32]=readWeights(1024,  gamma_file_7);
    general_parameter[33]=readWeights(1024,  mean_file_7);
    general_parameter[34]=readWeights(1024,  invstd_file_7);

    general_parameter[35]=readWeights(1024,  bias_file_8);
    general_parameter[36]=readWeights(1024,  beta_file_8);
    general_parameter[37]=readWeights(1024,  gamma_file_8);
    general_parameter[38]=readWeights(1024,  mean_file_8);
    general_parameter[39]=readWeights(1024,  invstd_file_8);

    general_parameter[40]=readWeights(10,  bias_file_9);
    general_parameter[41]=readWeights(10,  beta_file_9);
    general_parameter[42]=readWeights(10,  gamma_file_9);
    general_parameter[43]=readWeights(10,  mean_file_9);
    general_parameter[44]=readWeights(10,  invstd_file_9);

    // time spent for loading all trained parameters
    timer_0 = clock();

    // #######################################################################################
    // #######################################################################################
    // #############################  Start of Main program  #################################
    // #######################################################################################  
    // #######################################################################################

    // set the number of testing images 
    int test_num = IMAGE_NUM;
    int img_index;
    float accuracy=0;
    int correct_count=0;
    for (img_index=0; img_index<test_num; img_index++){

        int img_position = img_index;
       // read cifar data from binary file
        unsigned char buffer[3073];
        FILE *ptr;
        ptr = fopen("test_batch.bin","rb");  // r for read, b for binary
        fseek(ptr, sizeof(buffer)*(img_position-1),SEEK_CUR); // go to the position of specific image
        fread(buffer,sizeof(buffer),1,ptr); 

        // save cifar data to image structure
        struct cifar_img img_eg;
        img_eg.label_img=buffer[0];
        printf("the label of example image is :%d\n",img_eg.label_img);
        
        unsigned char zero='0';
        int temp_pixel;
        float std_pixel;

        int dim, j, k;
        for( dim=0; dim<IMAGE_DIMENSION; dim++)
        {
            for( i=0; i<IMAGE_ROW; i++)
            {
                for( j=0; j<IMAGE_COL; j++)
                {
                    // save values from binary to image struct
                    temp_pixel=buffer[ dim*1024 + 32*i +j + 1]-zero + 48;
                    // when input image data, change the scale from 0-255 to -1 - +1;
                    std_pixel = (temp_pixel*2.0/255.0) - 1.0;
                    img_eg.pixel_img[dim][i][j]=std_pixel;
                }
            }
        }

        // #############################  Start of CNN  #################################

        // input-layer is 32*32 RGB image
        // for start,the feature_map is just the cifar10-image
        for ( i=0; i<IMAGE_DIMENSION; i++){
            for ( j=0; j<IMAGE_ROW; j++){
                for( k=0; k<IMAGE_COL; k++){
                    ImageMap.mapping_values[i][j][k]=img_eg.pixel_img[i][j][k];
                }
            }
        }

        ImageMap.valid_dim_1=IMAGE_DIMENSION;
        ImageMap.valid_dim_2=IMAGE_ROW;
        ImageMap.valid_dim_3=IMAGE_COL;

        // # 128C3-128C3-P2    
        // Conv2DLayer(3, 128, 1);
        Conv2DLayer(3, 128, 1,  cnn_kernel_1, general_parameter[0]);
        // BatchNormLayer(1, 128, 'C');
        BatchNormLayer(128, 'C', general_parameter[1],general_parameter[2],general_parameter[3],general_parameter[4]);
        // BatchNormLayer(128, 'C', "arr_0", "arr_0", "arr_0", "arr_0");
        ActivationLayer('c', 't');

        Conv2DLayer(3, 128, 1,  cnn_kernel_2, general_parameter[5]);
        MaxPooling(2);
        BatchNormLayer( 128, 'C', general_parameter[6],general_parameter[7],general_parameter[8],general_parameter[9]);
        ActivationLayer('c', 't');

        // # 256C3-256C3-P2   
        Conv2DLayer(3, 256, 1,  cnn_kernel_3, general_parameter[10]);
        BatchNormLayer(256, 'C', general_parameter[11],general_parameter[12],general_parameter[13],general_parameter[14]);
        ActivationLayer('c', 't');

        Conv2DLayer(3, 256, 1,  cnn_kernel_4, general_parameter[15]);
        MaxPooling(2);
        BatchNormLayer( 256, 'C', general_parameter[16],general_parameter[17],general_parameter[18],general_parameter[19]);
        ActivationLayer('c', 't');

        // # 512C3-512C3-P2  
        Conv2DLayer(3, 512, 1,  cnn_kernel_5, general_parameter[20]);
        BatchNormLayer(512, 'C', general_parameter[21],general_parameter[22],general_parameter[23],general_parameter[24]);
        ActivationLayer('c', 't');

        Conv2DLayer(3, 512, 1,  cnn_kernel_6, general_parameter[25]);
        MaxPooling(2);
        BatchNormLayer( 512, 'C', general_parameter[26],general_parameter[27],general_parameter[28],general_parameter[29]);
        ActivationLayer('c', 't');

        ResizeMapping2List();

        // # 1024FP-1024FP-10FP     
        DenseLayer(1024, fullConectWeight_1, general_parameter[30]);
        BatchNormLayer( 1024, 'F', general_parameter[31],general_parameter[32],general_parameter[33],general_parameter[34]);
        ActivationLayer('d', 't');

        DenseLayer(1024, fullConectWeight_2, general_parameter[35]);
        BatchNormLayer(1024, 'F', general_parameter[36],general_parameter[37],general_parameter[38],general_parameter[39]);
        ActivationLayer('d','t');

        DenseLayer(10, fullConectWeight_3, general_parameter[40]);
        BatchNormLayer(10, 'F', general_parameter[41],general_parameter[42],general_parameter[43],general_parameter[44]);

        // get the output and find the estimated label
        int max_ind=1;
        float max_value=NNLayer.list_values[0];
        for ( i=0; i<10; i++){
            printf("%.5f\t", NNLayer.list_values[i]);
            if(NNLayer.list_values[i] >max_value ){
                max_ind = i;
                max_value =NNLayer.list_values[i];
            }
        }
        printf("The estimated label/ actual label is : %d\t %d \n", max_ind, img_eg.label_img);

        // counter update, to calculate the accuracy
        if (max_ind == img_eg.label_img){
            correct_count++;
            
        }
        printf("at %d image, count: %d\n", (img_index+1), correct_count);
        accuracy = correct_count/(img_index+1.0);
        printf("accuracy: %.3f\n\n", accuracy);
    }

    // time spent on CNN computation
    timer_end = clock();
    // print time 
    float t_0=(double)(timer_0-start)/CLOCKS_PER_SEC;
    float t_e=(double)(timer_end-start)/CLOCKS_PER_SEC;
    float t_a=t_e/IMAGE_NUM;

    printf("%.6f seconds for initialization (loading weights)\n", t_0);
    printf("%.6f seconds for whole classification process\n", t_e);
    printf("%.6f seconds for average classification time (1 image) \n", t_a);
    printf("So the final accuracy for %d images: %.3f\n", test_num, accuracy);

    return 0;
}

