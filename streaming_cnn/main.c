#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <inttypes.h>

#include "xtimer.h"


#define MAX_LINE_LENGTH 1024

typedef float real_t;
void print_array(const real_t* array, int size){
    for (int i = 0; i < size; i++) {
        printf("%.6f", array[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

real_t **allocate_2d_array(int rows, int cols){
    real_t ** array = malloc(rows * sizeof(real_t *));
    for (int i = 0; i< rows; i++){
        array[i] = malloc(cols * sizeof(real_t));
        for (int j = 0;j<cols;j++){
            array[i][j] = 0.0f;
        }
    }
    return array;
}

real_t*** allocate_3d_array(int first, int second, int third){
    real_t*** array = malloc(first * sizeof(real_t **));
    for (int i = 0; i< first; i++){
        array[i] = allocate_2d_array(second,third);
    }
    return array;
}

void free_2d_array(real_t **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void free_3d_array(real_t ***array, int first, int second) {
    for (int i = 0; i < first; i++) {
        free_2d_array(array[i], second);
    }
    free(array);
}



void conv1d_and_relu_multi_channel(real_t *input, real_t** kernel, real_t ** output, real_t* bias , int channel_number, int inputSize, int kernelSize){
    for (int channel = 0; channel < channel_number; channel++){
        for(int i = 0; i< inputSize - kernelSize + 1; i++){
            output[channel][i] = 0.0f;
            for(int j = 0; j< kernelSize; j++){
                output[channel][i] += input[i + j] * kernel[channel][j];
            }
            output[channel][i] += bias[channel];
            if (output[channel][i] <= 0){
                output[channel][i] = 0.0f;
            }
        }
    }
}



void fill_tile(real_t* tile, real_t* input_data, int i ,int tile_size){
    for (int j = 0; j < tile_size; j++ ){
        tile[j] = input_data[i + j];
    }
}

void maxpool1d_channel(real_t** tile, int tile_size, int channel_number, real_t** output_tile){
    for (int channel = 0; channel < channel_number; channel++){
        for (int i= 0; i< tile_size; i+=2){
            if (tile[channel][i]> tile[channel][i+1] ){
                output_tile[channel][i/2] = tile[channel][i];
                
            }
            else{
                output_tile[channel][i/2] = tile[channel][i+1];
            }
        }
    }
}

void multi_channel_aggregation_and_pooling(real_t** input_tile, real_t* output_tile, real_t*** kernel, int input_channels, int output_channels, int tile_size, int kernel_size, int position, int full_input_size){
    //This function performs the second convolution and the average pooling right after
    // The only problem right now is that I must take care des effets de bord.
    for(int channel = 0; channel < output_channels; channel++){
        for (int input_channel = 0; input_channel < input_channels; input_channel++){
            for(int pos = 0; pos < tile_size;pos ++){
                for (int kernel_nbr = 0; kernel_nbr < kernel_size; kernel_nbr++){
                    
                    if (position + pos - kernel_nbr >= 0 && position + pos - kernel_nbr + 3 <= full_input_size){
                        output_tile[channel] += input_tile[input_channel][pos] * kernel[channel][input_channel][kernel_nbr];
                    }
                }
            }
        }
    }
}

void mlp(real_t* input, real_t* output, int input_size, int hidden_size, int output_size, real_t** weight1, real_t** weight2,real_t* bias1, real_t* bias2){
    real_t intermediate[hidden_size];
    for (int i = 0; i < hidden_size; i++){
        intermediate[i] =0.0f;
        for (int j = 0; j < input_size; j++){
            intermediate[i] += input[j] * weight1[i][j];
        }
        intermediate[i] += bias1[i];

        //Apply ReLU
        if (intermediate[i]<0){
            intermediate[i] = 0.0f;
        }   
    }
    for(int i = 0; i< output_size; i++){
        output[i] = 0.0f;
        for (int j = 0; j< hidden_size; j++){
            output[i] += intermediate[j] * weight2[i][j];
        }
        output[i] += bias2[i];
    }
}
void CNN_model_inference(real_t* input_data, real_t* output ,real_t** kernel1, int channel_number1, int kernelSize1, real_t *** kernel2, int channel_number2, int kernelSize2, int tile_size, int input_size, real_t** weight1, real_t** weight2,real_t* fcbias1, real_t* fcbias2, real_t* convbias1, real_t* convbias2){
//    real_t tile[tile_size + kernelSize1 -1]; // take too much stack!
    real_t* tile = malloc(sizeof(real_t) * (tile_size + kernelSize1 -1));
    real_t**  intermediate = allocate_2d_array(channel_number1, tile_size);
    real_t** intermediate2 = allocate_2d_array(channel_number1, tile_size/2);
    
    real_t output_tile[channel_number2];
    for (int i = 0; i< channel_number2;i++){
        output_tile[i] = 0.0f;
    }

    
    int outputSize = (input_size -kernelSize1 +1)/2 - kernelSize2 + 1;
    for (int i = 0; i < input_size - kernelSize2 ; i+= tile_size){
//        printf("i=%d\n",i);
        fill_tile(tile, input_data, i, tile_size + kernelSize1 -1);
        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1,tile_size + kernelSize1 - 1, kernelSize1);
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        multi_channel_aggregation_and_pooling(intermediate2, output_tile, kernel2, channel_number1, channel_number2, tile_size/2, kernelSize2,i/2,(input_size -kernelSize1 +1)/2);
    }
    puts("ping!");
    free_2d_array(intermediate,channel_number1);
    free_2d_array(intermediate2,channel_number1);
    
    for(int i = 0; i< 32;i++){
        output_tile[i] /= outputSize;;
        // Since the bias is the same for every element of the same channel
        // It is added outputSize times to a channel, so we just have to add it once after division
        output_tile[i] += convbias2[i];
    }
    //print_array(output_tile,channel_number2);
    printf("%f test",weight1[0][0]);
    mlp(output_tile, output, 32, 64, 2, weight1, weight2, fcbias1, fcbias2 );
}
/*
I have to rework this function
Either I flash the file containing the weights and use the filesystem to access it on the MCU
Or I compile the program directly with the values included in it
void read_parameters_from_file(const char *file_name,
                               real_t conv1_weight[],
                               real_t conv1_bias[],
                               real_t conv2_weight[],
                               real_t conv2_bias[],
                               real_t fc1_weight[],
                               real_t fc1_bias[],
                               real_t fc2_weight[],
                               real_t fc2_bias[],
                               int conv1_size,
                               int conv2_size,
                               int convb1_size,
                               int convb2_size,
                               int fc1_size,
                               int fc2_size,
                               int fcb1_size,
                               int fcb2_size) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "conv1.weight")) {
            for (int i = 0; i < conv1_size; ++i) {
                fscanf(file, "%lf", &conv1_weight[i]);
            }
        }else if (strstr(line, "conv1.bias")) {
            for (int i = 0; i < convb1_size; ++i) {
                fscanf(file, "%lf", &conv1_bias[i]);
            } 
        }else if (strstr(line, "conv2.weight")) {
            for (int i = 0; i < conv2_size; ++i) {
                fscanf(file, "%lf", &conv2_weight[i]);
            }
        }else if (strstr(line, "conv2.bias")) {
            for (int i = 0; i < convb2_size; ++i) {
                fscanf(file, "%lf", &conv2_bias[i]);
            } 
        } else if (strstr(line, "fc1.weight")) {
            for (int i = 0; i < fc1_size; ++i) {
                fscanf(file, "%lf", &fc1_weight[i]);
            }
        } else if (strstr(line, "fc1.bias")) {
            for (int i = 0; i < fcb1_size; ++i) {
                fscanf(file, "%lf", &fc1_bias[i]);
            }
        } else if (strstr(line, "fc2.weight")) {
            for (int i = 0; i < fc2_size; ++i) {
                fscanf(file, "%lf", &fc2_weight[i]);
            }
        } else if (strstr(line, "fc2.bias")) {
            for (int i = 0; i < fcb2_size; ++i) {
                fscanf(file, "%lf", &fc2_bias[i]);
            }
        }
    }

    fclose(file);
}

void fill2d(int dim1, int dim2, real_t flattened_array[], real_t** result_array ){
    int index = 0;
    for (int i = 0; i < dim1;i++){
        for (int j = 0; j < dim2; j ++){
            result_array[i][j] = flattened_array[index];
            index++;
        }
    }
}

void fill3d(int dim1, int dim2, int dim3, real_t flattened_array[], real_t*** result_array){
    int index = 0;
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            for (int k = 0; k < dim3; k++){
                result_array[i][j][k] = flattened_array[index];
                index++;
            }
        }
    }

}
*/

static real_t input_data[16000];
int main(void){
    // Test
    puts("Début des hostilités");
    
    int channel_number1 = 16;
    int kernelSize1 = 3;
    int channel_number2 = 32;
    int kernelSize2 = 3;
    int tile_size = 1024;
    int input_size = 16000;
    
    
    
    real_t output[2];

    
    real_t fcbias1[64];
    real_t fcbias2[2];
    real_t convbias1[16];
    real_t convbias2[32];
    real_t**  kernel1 = allocate_2d_array(channel_number1, kernelSize1);
    real_t*** kernel2 = allocate_3d_array(channel_number2, channel_number1, kernelSize2);
    real_t** weight1 = allocate_2d_array(64, 32);
    real_t** weight2 = allocate_2d_array(2,64);
    /*
    real_t kernel1_array[channel_number1 * kernelSize1];
    real_t kernel2_array[channel_number1 * channel_number2 * kernelSize2];
    real_t weight1_array[32*64];
    real_t weight2_array[64*2];
    read_parameters_from_file(
        "test.txt",
        kernel1_array,
        convbias1,
        kernel2_array,
        convbias2,
        weight1_array,
        fcbias1,
        weight2_array,
        fcbias2,
        channel_number1 * kernelSize1,
        channel_number1 * channel_number2 * kernelSize2,
        16,
        32,
        32*64,
        64*2,
        16,
        32
        );
    

    

    fill2d(channel_number1, kernelSize1, kernel1_array, kernel1);
    fill2d(64,32,weight1_array,weight1);
    fill2d(2,64,weight2_array,weight2);
    fill3d(channel_number2,channel_number1,kernelSize2,kernel2_array, kernel2);
    */

    //On va donner des valeurs en utilisant les 1d arrays.


    
    // Donner des valeurs aux kernel et aux weights du mlp
    printf("Start to init the parameters \n");
    for (int i=0; i < channel_number1; i++){
        for (int j = 0; j< kernelSize1; j++){
//            printf("kernel1 i: %d, j: %d \n", i, j);
            kernel1[i][j] = (i+j+1)/100.0f;
        }
        for(int j = 0; j< channel_number2;j++){
            for (int k = 0; k < kernelSize2; k++){
//                printf("kernel2 i: %d, j: %d, k: %d \n", i, j, k);
                kernel2[j][i][k] = ((i+1)*(j+1)*(k+1))/100.0f;
            }
        }
    }
    
    for (int i = 0;i<64;i++){
        for (int j = 0;j<32;j++){
//            printf("weight1 i: %d, j: %d \n", i, j);
            weight1[i][j] = (i + j) /100.0f;
        }
        for(int j = 0;j<2;j++){
//            printf("weight2 i: %d, j: %d \n", i, j);
            weight2[j][i]    = (real_t)i / (((real_t)j+1.0f)*100.0f) ;
        }
    }
    
//    printf("%f test",weight1[0][0]);
    for (int i = 0; i< 64; i++){
        fcbias1[i] = i;
    }
    for (int i = 0; i< 2;i++){
        fcbias2[i] = i;
    }
    for (int i = 0; i< 16;i++){
        convbias1[i] = i;
    }
    for (int i = 0; i< 32;i++){
        convbias2[i] = i;
    }
    
   for (int i = 0; i< input_size;i++){
        input_data[i] = i/10000.0f;
    }   
    puts("Ok on lance l'inference");
    
    uint32_t inference_duration;
    inference_duration = xtimer_now_usec();
    CNN_model_inference(input_data, output, kernel1, channel_number1, kernelSize1, kernel2, channel_number2, kernelSize2, tile_size, input_size, weight1, weight2,fcbias1,fcbias2, convbias1, convbias2);
    inference_duration = xtimer_now_usec() - inference_duration;
    
    printf("\nLe résultat est : \n");
    print_array(output,2);
    printf("inference duration in usec: %" PRIu32 " \n", inference_duration);
    

}
