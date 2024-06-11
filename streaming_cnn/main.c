#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#define MAX_LINE_LENGTH 1024


void print_array(const double* array, int size){
    for (int i = 0; i < size; i++) {
        printf("%.6f", array[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

double **allocate_2d_array(int rows, int cols){
    double ** array = malloc(rows * sizeof(double *));
    for (int i = 0; i< rows; i++){
        array[i] = malloc(cols * sizeof(double));
        for (int j = 0;j<cols;j++){
            array[i][j] = 0.0f;
        }
    }
    return array;
}

double*** allocate_3d_array(int first, int second, int third){
    double*** array = malloc(first * sizeof(double **));
    for (int i = 0; i< first; i++){
        array[i] = allocate_2d_array(second,third);
    }
    return array;
}

void free_2d_array(double **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void free_3d_array(double ***array, int first, int second) {
    for (int i = 0; i < first; i++) {
        free_2d_array(array[i], second);
    }
    free(array);
}



void conv1d_and_relu_multi_channel(double *input, double** kernel, double ** output, double* bias , int channel_number, int inputSize, int kernelSize){
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



void fill_tile(double* tile, double* input_data, int i ,int tile_size){
    for (int j = 0; j < tile_size; j++ ){
        tile[j] = input_data[i + j];
    }
}

void maxpool1d_channel(double** tile, int tile_size, int channel_number, double** output_tile){
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

void multi_channel_aggregation_and_pooling(double** input_tile, double* output_tile, double*** kernel, int input_channels, int output_channels, int tile_size, int kernel_size, int position, int full_input_size){
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

void mlp(double* input, double* output, int input_size, int hidden_size, int output_size, double** weight1, double** weight2,double* bias1, double* bias2){
    double intermediate[hidden_size];
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
void CNN_model_inference(double* input_data, double* output ,double** kernel1, int channel_number1, int kernelSize1, double *** kernel2, int channel_number2, int kernelSize2, int tile_size, int input_size, double** weight1, double** weight2,double* fcbias1, double* fcbias2, double* convbias1, double* convbias2){
    double tile[tile_size + kernelSize1 -1];
    double**  intermediate = allocate_2d_array(channel_number1, tile_size);
    double** intermediate2 = allocate_2d_array(channel_number1, tile_size/2);
    double output_tile[channel_number2];
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
                               double conv1_weight[],
                               double conv1_bias[],
                               double conv2_weight[],
                               double conv2_bias[],
                               double fc1_weight[],
                               double fc1_bias[],
                               double fc2_weight[],
                               double fc2_bias[],
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

void fill2d(int dim1, int dim2, double flattened_array[], double** result_array ){
    int index = 0;
    for (int i = 0; i < dim1;i++){
        for (int j = 0; j < dim2; j ++){
            result_array[i][j] = flattened_array[index];
            index++;
        }
    }
}

void fill3d(int dim1, int dim2, int dim3, double flattened_array[], double*** result_array){
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

static double input_data[16000];
int main(void){
    // Test
    puts("Début des hostilités");
    
    int channel_number1 = 16;
    int kernelSize1 = 3;
    int channel_number2 = 32;
    int kernelSize2 = 3;
    int tile_size = 1024;
    int input_size = 16000;
    
    
    
    double output[2];

    
    double fcbias1[64];
    double fcbias2[2];
    double convbias1[16];
    double convbias2[32];
    double**  kernel1 = allocate_2d_array(channel_number1, kernelSize1);
    double*** kernel2 = allocate_3d_array(channel_number2, channel_number1, kernelSize2);
    double** weight1 = allocate_2d_array(64, 32);
    double** weight2 = allocate_2d_array(2,64);
    /*
    double kernel1_array[channel_number1 * kernelSize1];
    double kernel2_array[channel_number1 * channel_number2 * kernelSize2];
    double weight1_array[32*64];
    double weight2_array[64*2];
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
    
    for (int i=0; i < channel_number1; i++){
        for (int j = 0; j< kernelSize1; j++){
            kernel1[i][j] = (i+j+1)/100.0f;;
        }
        for(int j = 0; j< channel_number2;j++){
            for (int k = 0; k < kernelSize2; k++){
                kernel2[j][i][k] = ((i+1)*(j+1)*(k+1))/100.0f;
            }
        }
    }
    
    for (int i = 0;i<64;i++){
        puts("a");
        for (int j = 0;j<32;j++){
            weight1[i][j] = (i + j) /100.0f;
        }
        for(int j = 0;j<2;j++){
            weight2[j][i]    = (double)i / (((double)j+1.0f)*100.0f) ;
        }
    }
    
    printf("%f test",weight1[0][0]);
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
    CNN_model_inference(input_data, output, kernel1, channel_number1, kernelSize1, kernel2, channel_number2, kernelSize2, tile_size, input_size, weight1, weight2,fcbias1,fcbias2, convbias1, convbias2);
    
    printf("\nLe résultat est : \n");
    print_array(output,2);
    

}
