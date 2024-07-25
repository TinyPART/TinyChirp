#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sndfile.h>
#include <dirent.h>
#include "array_data.h"
#include "main.h"
#define MAX_LINE_LENGTH 1024
#define MAX_PATH 1024

void processAudioFile( const char *filePath, int* confusion_matrix, int actual_class){
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(filePath, SFM_READ, &sfinfo);

    if (!infile){
        printf("Failed to open %s\n",filePath);
        return;
    }

    int numItems = sfinfo.frames * sfinfo.channels;
    float* buffer = (float *)malloc(numItems * sizeof(float));

    sf_read_float(infile, buffer, numItems);
    float output[2];
    int channel_number1 = 16;
    int kernelSize1 = 3;
    int channel_number2 = 32;
    int kernelSize2 = 3;
    int tile_size = 128;
    int input_size=numItems;
    CNN_model_inference(buffer, output, conv1weight, channel_number1, kernelSize1, conv2weight, channel_number2, kernelSize2, tile_size, input_size, fc1weight, fc2weight,fc1bias,fc2bias, conv1bias, conv2bias);
    
    if (output[0] > output[1]){
        confusion_matrix[actual_class * 2 ]++;
    }
    else{
        confusion_matrix[actual_class * 2 + 1]++;
    }

    sf_close(infile);

}


void processDirectory(char* dirPath, int* confusion_matrix, int actual_class){
    struct dirent *entry;
    DIR *dp = opendir(dirPath);


    if (!dp){
        perror("opendir");
        return;
    }
    printf("Ok\n");
    while((entry = readdir(dp))){

        if (strstr(entry->d_name, "Zone.Identifier") != NULL) {
            continue;
        }
        char filePath[MAX_PATH];
        snprintf(filePath, sizeof(filePath), "%s/%s", dirPath, entry->d_name);
        if (entry->d_type == DT_REG){
            processAudioFile(filePath,confusion_matrix,actual_class);
        }
    }

}



void print_array(const float* array, size_t size){
    for (size_t i = 0; i < size; i++) {
        printf("%.6f", array[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

float **allocate_2d_array(int rows, int cols){
    float ** array = malloc(rows * sizeof(float *));
    for (int i = 0; i< rows; i++){
        array[i] = malloc(cols * sizeof(float));
        for (int j = 0;j<cols;j++){
            array[i][j] = 0.0f;
        }
    }
    return array;
}

void conv1d_and_relu_multi_channel(float *input, float** kernel, float ** output, float* bias , int channel_number, int inputSize, int kernelSize){
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



void fill_tile(float* tile, float* input_data, int i ,size_t tile_size){
    for (int j = 0; j < tile_size; j++ ){
        tile[j] = input_data[i + j];
    }
}

void maxpool1d_channel(float** tile, int tile_size, int channel_number, float** output_tile){
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


void multi_channel_aggregation_and_pooling(float** input_tile, float* output_tile, float*** kernel, int input_channels, int output_channels, int tile_size, int kernel_size, int position, int full_input_size){
    //This function performs the second convolution and the average pooling right after
    // The only problem right now is that I must take care des effets de bord.
    for(int channel = 0; channel < output_channels; channel++){
        for (int input_channel = 0; input_channel < input_channels; input_channel++){
            for(int pos = 0; pos < tile_size;pos ++){
                for (int kernel_nbr = 0; kernel_nbr < kernel_size; kernel_nbr++){
                    
                    //printf("Position : %d, pos : %d , kernel_nbr : %d\n",position, pos, kernel_nbr);
                    //printf("Premier %d, Deuxieme %d, borne %d\n", position + pos - kernel_nbr,position + pos - kernel_nbr+3,full_input_size);
                    if (position + pos - kernel_nbr >= 0 && position + pos - kernel_nbr + 3 <= full_input_size){
                        //printf("pos %d kernel_nbr %d\n",pos,kernel_nbr);
                        
                        output_tile[channel] += input_tile[input_channel][pos] * kernel[channel][input_channel][kernel_nbr];
                        /*
                        if (channel == 0 && input_channel == 1){
                            printf("%f, %f\n",input_tile[input_channel][pos],kernel[channel][input_channel][kernel_nbr]);
                        }
                        */
                        
                    }
                }
            }
        }
    }
}

void mlp(float* input, float* output, int input_size, int hidden_size, int output_size, float** weight1, float** weight2,float* bias1, float* bias2){
    float intermediate[hidden_size];
    
    for (int i = 0; i < hidden_size; i++){
        intermediate[i] = 0.0f;
        for (int j = 0; j < input_size; j++){
            intermediate[i] += input[j] * weight1[i][j];
        }
        intermediate[i] += bias1[i];
        if (intermediate[i]<0){
            intermediate[i] = 0.0f;
        }   
    }
    for(int i = 0; i< 2; i++){
        output[i] = 0.0f;
        for (int j = 0; j< hidden_size; j++){
            output[i] += intermediate[j] * weight2[i][j];
        }
        output[i] += bias2[i];
    }
}



void CNN_model_inference(float* input_data, float* output ,float** kernel1, int channel_number1, size_t kernelSize1, float *** kernel2, int channel_number2, size_t kernelSize2, int tile_size, size_t input_size, float** weight1, float** weight2,float* fcbias1, float* fcbias2, float* convbias1, float* convbias2){
    float tile[tile_size + kernelSize1 -1];
    float**  intermediate = allocate_2d_array(channel_number1, tile_size);
    float** intermediate2 = allocate_2d_array(channel_number1, tile_size/2);
    float output_tile[channel_number2];
    for (int i = 0; i< channel_number2;i++){
        output_tile[i] = 0.0f;
    }
    int outputSize = (input_size -kernelSize1 +1)/2 - kernelSize2 + 1;
    int count = 0;
    for (int i = 0; i < input_size - kernelSize2 ; i+= tile_size){
        fill_tile(tile, input_data, i, tile_size + kernelSize1 -1);
        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1,tile_size + kernelSize1 - 1, kernelSize1);
        // Should have a tile with all the differents channels now
        // Must add the handling of multiple channels for the next layer
        
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        multi_channel_aggregation_and_pooling(intermediate2, output_tile, kernel2, channel_number1, channel_number2, tile_size/2, kernelSize2,i/2,(input_size -kernelSize1 +1)/2);
    }
    for(int i = 0; i< channel_number2;i++){
        output_tile[i] /= outputSize;
        output_tile[i] += convbias2[i];
    }
    // Jusque là on est bons
    mlp(output_tile, output, 32, 64, 2, weight1, weight2, fcbias1, fcbias2 );
}

float input_data[16000];
int main(void){
    const char* target = "target";
    const char* non_target = "non_target";
    int confusion_matrix[4];
    confusion_matrix[0] = 0;
    confusion_matrix[1] = 0;
    confusion_matrix[2] = 0;
    confusion_matrix[3] = 0;
    processDirectory(target, confusion_matrix, 0);
    processDirectory(non_target, confusion_matrix, 1);

    printf("%d %d %d %d\n",confusion_matrix[0], confusion_matrix[1], confusion_matrix[2], confusion_matrix[3]);



    
    

}
