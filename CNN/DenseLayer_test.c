#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define KERNEL_SIZE 3

// definition of image input
#define IMAGE_ROW 32
#define IMAGE_COL 32
#define IMAGE_DIMENSION 3


struct cifar_img{
    double pixel_img[IMAGE_DIMENSION][IMAGE_ROW][IMAGE_COL];
    int label_img;
};
// convolution kernel structure
// struct conv_kernel{
//     int kernel_dim;
//     double kernel_para[KERNEL_SIZE][KERNEL_SIZE];   
// };

// // image patch structure
// struct img_patch{
// 	double value[KERNEL_SIZE][KERNEL_SIZE];
// };

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

// double ConvolutionFunc(struct conv_kernel input_kernel, struct img_patch image_patch){
// 	int kernel_size;
// 	kernel_size=input_kernel.kernel_dim;
//     // do multiplication-addition for a single window
//     double convolution_result=0;
//     for (int m=0; m<kernel_size; m++){
//         for (int n=0; n<kernel_size; n++){
//             convolution_result = input_kernel.kernel_para[m][n]* image_patch.value[m][n] + convolution_result;
//         }
//     }


// 	return convolution_result;
// }

#define MAX_DIMENSION_1 512*2
#define MAX_DIMENSION_2 32*2
#define MAX_DIMENSION_3 32*2

#define ARRAY_LENGTH 32*2

static struct image_map {
    int valid_dim_1;
    int valid_dim_2;
    int valid_dim_3;
    double mapping_values[MAX_DIMENSION_1][MAX_DIMENSION_2][MAX_DIMENSION_3];
}ImageMap;


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
                printf("%.f\t",img_eg.pixel_img[dim][i][j]);
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


    // do maxpooling for pool_size=8
    MaxPooling(8);
    MaxPooling(2);


    for(int dim=0; dim<ImageMap.valid_dim_1; dim++)
    {
        for(int i=0; i<ImageMap.valid_dim_2; i++)
        {
            for(int j=0; j<ImageMap.valid_dim_3; j++)
            {
                // print image to the terminal
                printf("%.f\t",ImageMap.mapping_values[dim][i][j]);
            }
            printf("\n");

        }

        printf("\n\n");

    }

    // implement activation
    ResizeMapping2List();

    for(int dim=0; dim<NNLayer.valid_list_index; dim++)
    {
    	printf("%.f\t",NNLayer.list_values[dim]);
    	
    }
    printf("\n");


    // implement DenseLayer (all weight is 1)
    DenseLayer(10);

    for(int dim=0; dim<NNLayer.valid_list_index; dim++)
    {
    	printf("%.f\t",NNLayer.list_values[dim]);
    	
    }
    printf("\n");



	
	return 0;
}