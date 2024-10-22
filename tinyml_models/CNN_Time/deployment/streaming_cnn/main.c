#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <inttypes.h>
#include "xtimer.h"
#include "array_data.h"

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

#define CHANNEL_NUM1 4
#define CHANNEL_NUM2 8
#define KERNEL_SIZE1 3
#define KERNEL_SIZE2 3
#define TILE_SIZE 128
#define INPUT_SIZE 16000

static real_t tile[TILE_SIZE + KERNEL_SIZE1 -1];
static real_t intermediate_val[CHANNEL_NUM1][TILE_SIZE];
static real_t intermediate2_val[CHANNEL_NUM1][TILE_SIZE/2];


void CNN_model_inference(real_t* input_data, real_t* output ,real_t** kernel1, int channel_number1, int kernelSize1, real_t *** kernel2, int channel_number2, int kernelSize2, int tile_size, int input_size, real_t** weight1, real_t** weight2,real_t* fcbias1, real_t* fcbias2, real_t* convbias1, real_t* convbias2){
//    real_t tile[tile_size + kernelSize1 -1]; // take too much stack!
    real_t*  intermediate[CHANNEL_NUM1];
    real_t* intermediate2[CHANNEL_NUM1];
    
    for (int i = 0; i < CHANNEL_NUM1; i++) {
        intermediate[i] = &intermediate_val[i][0];
        intermediate2[i] = &intermediate2_val[i][0];
    }
    
    real_t output_tile[channel_number2];
    for (int i = 0; i< channel_number2;i++){
        output_tile[i] = 0.0f;
    }

    int outputSize = (input_size -kernelSize1 +1)/2 - kernelSize2 + 1;
    for (int i = 0; i < input_size - kernelSize2 ; i+= tile_size){

        fill_tile(tile, input_data, i, tile_size + kernelSize1 -1);
        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1,tile_size + kernelSize1 - 1, kernelSize1);
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        multi_channel_aggregation_and_pooling(intermediate2, output_tile, kernel2, channel_number1, channel_number2, tile_size/2, kernelSize2,i/2,(input_size -kernelSize1 +1)/2);
    }

    
    for(int i = 0; i< channel_number2;i++){
        output_tile[i] /= outputSize;;
        // Since the bias is the same for every element of the same channel
        // It is added outputSize times to a channel, so we just have to add it once after division
        output_tile[i] += convbias2[i];
    }

    mlp(output_tile, output, channel_number2, 64, 2, weight1, weight2, fcbias1, fcbias2 );
}


static real_t input_data[16000];
int main(void){
    // Test
    
    int channel_number1 = CHANNEL_NUM1;
    int kernelSize1 = KERNEL_SIZE1;
    int channel_number2 = CHANNEL_NUM2;
    int kernelSize2 = KERNEL_SIZE2;
    int tile_size = TILE_SIZE;
    int input_size = INPUT_SIZE;
    
    for (int i = 0; i < 16000;i++){
        input_data[i] = i/16000.0f;
    }
    
    real_t output[2];
    
    while(1) {
        uint32_t inference_duration;
        inference_duration = xtimer_now_usec();
        
        for (int i = 0; i < 3; i++) {
            CNN_model_inference(input_data, output, conv1weight, channel_number1, kernelSize1, conv2weight, channel_number2, kernelSize2, tile_size, input_size, fc1weight, fc2weight,fc1bias,fc2bias, conv1bias, conv2bias);
        }
        inference_duration = xtimer_now_usec() - inference_duration;
        printf("inference duration in usec: %" PRIu32 " \n", inference_duration);
    }
    
    printf("Inference output : \n");
    print_array(output,2);
    
    return 0; 
}
