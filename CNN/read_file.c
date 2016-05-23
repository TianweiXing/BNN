#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// definition of image input
#define IMAGE_ROW 32
#define IMAGE_COL 32
#define IMAGE_DIMENSION 3

struct cifar_img{
    double pixel_img[IMAGE_DIMENSION][IMAGE_ROW][IMAGE_COL];
    int label_img;
};

int main(){

    // the number of input images
    int input_number = 5;

    unsigned char buffer[3073*input_number];
    FILE *ptr;

    ptr = fopen("data_batch_1.bin","rb");  // r for read, b for binary

    
    // fread(buffer,sizeof(buffer),1,ptr); // read 10 bytes to our buffer
    fread(buffer,3073,input_number,ptr);

//You said you can read it, but it's not outputting correctly... 
    // keep in mind that when you "output" this data, you're not reading ASCII, 
    //so it's not like printing a string to the screen:
    // for(int i = 0; i<3073; i++)
    // {
    //     // printf("%u ", buffer[i]); // prints a series of bytes
    // }
    // fclose(ptr);

    struct cifar_img img_eg[input_number];
    int i;
    for (i=0; i<input_number; i++){
        img_eg[i].label_img=buffer[i*3073];
        printf("%u\n",buffer[i*3073]);
        printf("the label of this example image is :%d\n",img_eg[i].label_img);

        int dim, m, n;
        double temp_pixel;
        unsigned char zero='0';

        for(int dim=0; dim<IMAGE_DIMENSION; dim++)
       {
            for(int m=0; m<IMAGE_ROW; m++)
            {
                for(int n=0; n<IMAGE_COL; n++)
                {
                    // save values from binary to image struct
                    temp_pixel=buffer[ i*3073 + dim*1024 + 32*m +n + 1]-zero + 48;
                    img_eg[i].pixel_img[dim][m][n]=temp_pixel;
                    // print image to the terminal
                    printf("%.f\t",img_eg[i].pixel_img[dim][m][n]);
                }
                printf("\n");

            }
            printf("\n\n");

        }


    }

    return 0;
}