/*
 * Copyright (C) 2023 Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     apps
 * @{
 *
 * @file
 * @brief       Transformer Audio Application
 *
 * @author      Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * @}
 */

#include <stdio.h>
#include <string.h>
#include "xtimer.h"
#include "random.h"
#include "stdio_base.h"
#include "mlmci.h"
#include "conv_array_data.h"
#include <tvmgen_default.h>
extern mlmodel_t *model_ptr;
typedef float real_t;

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

void free_2d_array(real_t **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
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

void adpool_cumulative(real_t* output, real_t** multi_channel_tile, int tile_size, int output_size){
    for (int i = 0;i<output_size;i++){
        for( int j = 0;j<tile_size;j++){
            output[i] += multi_channel_tile[i][j];
        }
    }
    return;
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
void convolutional_layers_inference(real_t* input_data, real_t* output ,real_t** kernel1, int channel_number1, int kernelSize1, int tile_size, int input_size, real_t* convbias1){
//    real_t tile[tile_size + kernelSize1 -1]; // take too much stack!
    real_t* tile = malloc(sizeof(real_t) * (tile_size + kernelSize1 -1));
    real_t**  intermediate = allocate_2d_array(channel_number1, tile_size);
    real_t** intermediate2 = allocate_2d_array(channel_number1, tile_size/2);

    int outputSize = (input_size -kernelSize1 +1)/2;
    for (int i = 0; i <= input_size - kernelSize1; i+= tile_size){
        fill_tile(tile, input_data, i, tile_size + kernelSize1 -1);
        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1,tile_size + kernelSize1 - 1, kernelSize1);
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        adpool_cumulative(output, intermediate2, tile_size/2, channel_number1);
    }
    for (int i = 0;i<channel_number1;i++){
        output[i] /= outputSize;
    }
}
static real_t input_data[16000];
void model_inference(void)
{       


        //set input data
        mlmodel_iovar_t *input = mlmodel_get_input_variable(model_ptr, 0);

        int channel_number1 = 16;
        int kernelSize1 = 3;
        int tile_size = 128;
        int input_size = 16000;
        for (int i = 0; i < 16000;i++){
        input_data[i] = i/160.0f;
        }

        //run inference
        int start, end;
        start =  xtimer_now_usec();
        for(int i = 0; i < 3; i++){
        convolutional_layers_inference(input_data,input->values,conv1weight,channel_number1,kernelSize1,tile_size,input_size,conv1bias);
        }   
        
        //float* input_val = input->values; //pointer to input data
        
        int ret_val = mlmodel_inference(model_ptr);
        end =  xtimer_now_usec();
        printf("inference usec: %ld, ret: %d \n", (long int)(end - start), ret_val);
        
        //fetch model output
        mlmodel_iovar_t *output = mlmodel_get_output_variable(model_ptr, 0);
        float* output_val = output->values; //pointer to output data
}


int main(void)
{
    mlmodel_init(model_ptr);
    mlmodel_set_global_model(model_ptr);

    model_inference();

    return 0;
}
