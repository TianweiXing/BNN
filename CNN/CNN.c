#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// definition of image input
#define IMAGE_ROW 32
#define IMAGE_COL 32
#define IMAGE_DIMENSION 3
//  image_dimension repressent the 3 dimensional RGB value

// define array length for ordinary array use
#define ARRAY_LENGTH 10

// definition of convolution kernels
#define KERNEL_SIZE 3
//  image_dimension repressent the 3 dimensional RGB value

// image structure
struct cifar_img{
    double pixel_img[IMAGE_DIMENSION][IMAGE_ROW][IMAGE_COL];
    int label_img;
};

// convolution kernel structure
struct conv_kernel{
    int kernel_dim;
    double kernel_para[KERNEL_SIZE][KERNEL_SIZE];
    
};

// image patch structure
struct img_patch{
    double value[KERNEL_SIZE][KERNEL_SIZE];
};

// generate random number from -1 to 1
double getRand(){
    return (( (double)rand() * 2 ) / ( (double)RAND_MAX + 1 ) ) - 1;
}


// function for load kernel parameters 
// use random generated initial parameters: set ifRand=1 (by default)
struct conv_kernel LoadKernel(struct conv_kernel input_kernel, int init_kernel=1 ){
    // to set the kernel values from txt parameter files 
    // exported from python CNN model

    struct conv_kernel set_kernel;

    int i, j;
    if(init_kernel){

        // set the random values for every kernel
        set_kernel=input_kernel;
        // i is the kernel row            
        for(i=0; i<set_kernel.kernel_dim; i++){
            // j is the kernel col
            for(j=0; j<set_kernel.kernel_dim; j++){
                    set_kernel.kernel_para[i][j]=getRand();
            }
        }


    }
    return set_kernel;
}


// avoid making dynamic structure
// struct image_structure{
//     int row_number;
//     int col_number;
//     int image_3d;
//     double pixel_img[][][];
// };

// use a static struct instead
#define MAX_DIMENSION_1 512*2
#define MAX_DIMENSION_2 32*2
#define MAX_DIMENSION_3 32*2

static struct image_map {
    int valid_dim_1;
    int valid_dim_2;
    int valid_dim_3;
    double mapping_values[MAX_DIMENSION_1][MAX_DIMENSION_2][MAX_DIMENSION_3];
}ImageMap;

// initialization for ImageMap in the Main function


// Function for convolutional operation
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


// function for convolution layer
// unable to test it yet (question about paper)
void Conv2DLayer(struct conv_kernel input_kernel[], int num_kernel, int pad, int binary=0){
    int actual_dim1=ImageMap.valid_dim_1;
    int actual_dim2=ImageMap.valid_dim_2 + 2 * pad;
    int actual_dim3=ImageMap.valid_dim_3 + 2 * pad;

    // update ImageMap valid dimensions
    ImageMap.valid_dim_1= num_kernel * actual_dim1;
    ImageMap.valid_dim_2= actual_dim2 -(input_kernel[0].kernel_dim-1);
    ImageMap.valid_dim_3= actual_dim3 -(input_kernel[0].kernel_dim-1);

    // find out if it's necessary???
    double image_3d[actual_dim1][actual_dim2][actual_dim3];

    if(binary){
        printf("using binary kernel to do convolution");
        // change the kernels to binary kernels
    }


    // do convolution for every single kernel
    for(int i=0; i<num_kernel; i++){

        int kernel_size = input_kernel[i].kernel_dim;

        // for every slice of the image map
        for(int j=0; j<actual_dim1; j++){

            double image_slice[actual_dim2][actual_dim3];

            // initialize image slice using image_map stored value
            for(int m=0; m<actual_dim2; m++){
                for (int n=0; n<actual_dim3; n++){
                    // if on the pad edge, set the value to 0
                    if ((m==0)|(n==0)) {
                        image_slice[m][n] = 0;
                    }
                    else{
                        image_slice[m][n] = ImageMap.mapping_values[actual_dim1][m-1][n-1];
                    }

                }
            }

            // create result image slice 
            // currently we are not considering stride here: default stride is 1;
            int result_dim2 = actual_dim2 -(kernel_size-1);
            int result_dim3 = actual_dim3 -(kernel_size-1);
            double result_image[result_dim2][result_dim3];

            for(int k=0; k<result_dim2; k++){
                for(int l=0; l<result_dim3; l++){
                    
                    struct img_patch image_patch;

                    // set each image_patch
                    for(int m=0; m<kernel_size; m++){
                        for(int n=0; n<kernel_size; n++){
                            image_patch.value[m][n]= image_slice[k+m][l+n];
                        }
                    }

                    // calculate the convolution for this window
                    result_image[k][l]= ConvolutionFunc(input_kernel[i], image_patch);
                    // printf("%.f\t",result_image[i][j]);
                }
                // printf("\n");
            }
            

            // update the mapping values of ImageMap
            for (int m=0; m<ImageMap.valid_dim_2; m++){
                for (int n=0; n<ImageMap.valid_dim_3; n++){
                    ImageMap.mapping_values[actual_dim1 * i + j][m][n]=result_image[m][n];
                }
            }

        // for every actual dimension
        }
    // for every kernel
    }

}

// function for finding the maximun number in an 1D array
// function for finding the maximun number in an 1D array
double MaxNum(double array[], int length){
    double max_number=array[0];
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
        double pooled_image[new_dim2][new_dim2];
        double pooled_array[ARRAY_LENGTH];

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
void ActivationLayer(char activation, int binary=0){

    for(int dim=0; dim<ImageMap.valid_dim_1; dim++)
    {
        for(int i=0; i<ImageMap.valid_dim_2; i++)
        {
            for(int j=0; j<ImageMap.valid_dim_3; j++)
            {   
                double temp;
                temp = ImageMap.mapping_values[dim][i][j];
                // for different activation
                // use binary activation in binary mode
                switch (activation){
                    case 't': 
                    // hyper tangent function 
                        if (binary){
                            printf("write binary activation function here");
                            ImageMap.mapping_values[dim][i][j]=( 2.0 / ( 1 + exp( temp ) ) ) - 1;
                        }
                        else{

                            ImageMap.mapping_values[dim][i][j]=( 2.0 / ( 1 + exp( temp ) ) ) - 1;

                        }
                        break;
                    default:
                        break;
                }

            }
        }
    }
}


void BatchNormLayer(){
}


// use a static struct for fully connected layers
#define MAX_NEURON_NUM 10000

static struct NNLayer_list {
    int valid_list_index;
    double list_values[MAX_NEURON_NUM];
}NNLayer;


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
void DenseLayer(int output_node_num,  int binary=0, int init_weight=1){

    int input_node_num;
    input_node_num = NNLayer.valid_list_index;
    double weight_array[output_node_num][input_node_num];

    // load weight parameters from file
    int i, j;
    if (init_weight){

        for (i=0; i<output_node_num; i++){
            for (j=0; j<input_node_num; j++){

                weight_array[i][j] = 1;
            }
        }
    }

    // creating a new list to store the output
    double output_values[output_node_num];

    // calculating the weighted sum value
    for (i=0; i<output_node_num; i++){

        int weighted_sum =0;
        for (j=0; j<input_node_num; j++){

            weighted_sum = weighted_sum + NNLayer.list_values[j] * weight_array[i][j];
        }

        output_values[i] = weighted_sum;
    }

    // store output_values to NNLayer
    NNLayer.valid_list_index = output_node_num;

    for (i=0; i<output_node_num; i++){

        NNLayer.list_values[i] = output_values[i];
    }
}



int main(){

    // read cifar data from binary file

    unsigned char buffer[3073];
    FILE *ptr;

    ptr = fopen("data_batch_1.bin","rb");  // r for read, b for binary
    // int aaa;
    // aaa=sizeof(buffer);
    // printf("%d\n",aaa);

    fread(buffer,sizeof(buffer),1,ptr); // read 10 bytes to our buffer

    //You said you can read it, but it's not outputting correctly... 
    // keep in mind that when you "output" this data, you're not reading ASCII, 
    //so it's not like printing a string to the screen:
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

    for(int dim=0; dim<IMAGE_DIMENSION; dim++)
    {
        for(int i=0; i<IMAGE_ROW; i++)
        {
            for(int j=0; j<IMAGE_COL; j++)
            {
                // save values from binary to image struct
                temp_pixel=buffer[ dim*1024 + 32*i +j + 1]-zero + 48;
                img_eg.pixel_img[dim][i][j]=temp_pixel;
                // print image to the terminal
                // printf("%.f\t",img_eg.pixel_img[dim][i][j]);
            }
            // printf("\n");

        }

        // printf("\n\n");

    }
    // for test
    // int temp, temp2;
    // temp=buffer[1]-'0'+48;
    // printf("%d\n",temp);
    // temp2=buffer[2]-'0'+48;
    // printf("%d\n",temp2);
    // printf("%u\n",buffer[1]);
    

    // kernel function test
    // the max_kernel_size is set to 3, so here the dimension should be smaller than 3
    struct conv_kernel kernel_eg[2];
    kernel_eg[0].kernel_dim=3;
    kernel_eg[1].kernel_dim=2;

    for(int i=0; i<2; i++){
        kernel_eg[i]=LoadKernel(kernel_eg[i], 1);

        for(int j=0; j<kernel_eg[i].kernel_dim; j++){
            for(int k=0; k<kernel_eg[i].kernel_dim; k++){
                printf("%f\t",kernel_eg[i].kernel_para[j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
    

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


    // test conv2d layer
    // kernel_numuber=2;
    // Conv2DLayer(kernel_eg, kernel_numuber, 1, binary=0);


    
    // example of creating kernels, loading kernels, convolution, maxpooling and Nonlinear activation;
    struct conv_kernel conv_128_1[128];

    for(int i=0; i<128; i++){
        conv_128_1[i].kernel_dim=3;
        conv_128_1[i]=LoadKernel(conv_128_1[i], 1);

        for(int j=0; j<conv_128_1[i].kernel_dim; j++){
            for(int k=0; k<conv_128_1[i].kernel_dim; k++){
                // printf("%f\t",conv_128_1[i].kernel_para[j][k]);
            }
            // printf("\n");
        }
        // printf("\n\n");
    }

    Conv2DLayer(conv_128_1, 128, 1);
    printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);
    printf("\n");

    MaxPooling(2);
    printf("%d\t%d\t%d\t",ImageMap.valid_dim_1,ImageMap.valid_dim_2,ImageMap.valid_dim_3);

    ActivationLayer('t');





    // creat convolutional neural network

//    creat_net();

//    Feed_img_to_net();

//    get_output();

//    result_analysis();

    return 0;
}