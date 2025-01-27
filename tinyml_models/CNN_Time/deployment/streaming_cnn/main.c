#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <inttypes.h>
#include "xtimer.h"
#include "ztimer.h"
#include "blob/demo_samples_binary/target/t_audio_Corn_Bunting1.bin.h"
#include "blob/demo_samples_binary/three_segments_of_ntb_chiffchaff/ntb_chiffchaff_segment1.bin.h"
#include "blob/demo_samples_binary/three_segments_of_ntb_chiffchaff/ntb_chiffchaff_segment2.bin.h"
#include "blob/demo_samples_binary/three_segments_of_ntb_chiffchaff/ntb_chiffchaff_segment3.bin.h"

#include "periph/adc.h"
#include "periph/gpio.h"
#include "board.h"


#define RES             ADC_RES_10BIT
#define DELAY_US        62 //
#define ADC_BITS 10
#define GPIO_OUT_HIGHDRIVE GPIO_MODE(1, 1, 0, 3)

const float ADC_REF_V = 3.3 / 4;

#define PORT_BIT            (1 << 5)
#define PIN_MASK            (0x1f)

/* Compatibility wrapper defines for nRF9160 */
#ifdef NRF_P0_S
#define NRF_P0 NRF_P0_S
#endif

#ifdef NRF_P1_S
#define NRF_P1 NRF_P1_S
#endif
/**
 * @brief   Get the port's base address
 */
static inline NRF_GPIO_Type *port(gpio_t pin)
{
#if (CPU_FAM_NRF51)
    (void) pin;
    return NRF_GPIO;
#elif defined(NRF_P1)
    return (pin & PORT_BIT) ? NRF_P1 : NRF_P0;
#else
    (void) pin;
    return NRF_P0;
#endif
}

/**
 * @brief   Get a pin's offset
 */
static inline int pin_num(gpio_t pin)
{
#if GPIO_COUNT > 1
    return (pin & PIN_MASK);
#else
    return (int)pin;
#endif
}

int gpio_init_out_highdrive(gpio_t pin)
{

    port(pin)->PIN_CNF[pin_num(pin)] = GPIO_OUT_HIGHDRIVE;
    return 0;
}

void print_buffer_details(const float *buffer, size_t size) {
    printf("Buffer size: %u\n", size);
    printf("First 10 values:\n");
    for (size_t i = 0; i < 10; i++) {
        printf("ring_buffer[%u] = %.6f\n", i, buffer[i]);
    }
}

#define MAX_LINE_LENGTH 1024

typedef float real_t;
#include "array_data.h"
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
#define ACTUAL_TILE_SIZE (TILE_SIZE + KERNEL_SIZE1 -1)

static real_t tile[ACTUAL_TILE_SIZE];
static real_t intermediate_val[CHANNEL_NUM1][TILE_SIZE];
static real_t intermediate2_val[CHANNEL_NUM1][TILE_SIZE/2];



void print_array_output_tile(const float *array, size_t size) {
    printf("Output Tile:\n");
    for (size_t i = 0; i < size; i++) {
        printf("output_tile[%u] = %.6f\n", i, array[i]);
    }
}

void CNN_model_inference(real_t* input_data, real_t* output ,real_t** kernel1, int channel_number1, int kernelSize1, real_t *** kernel2, int channel_number2, int kernelSize2, int tile_size, int input_size, real_t** weight1, real_t** weight2,real_t* fcbias1, real_t* fcbias2, real_t* convbias1, real_t* convbias2, real_t* output_tile){
//    real_t tile[tile_size + kernelSize1 -1]; // take too much stack!
    real_t*  intermediate[CHANNEL_NUM1];
    real_t* intermediate2[CHANNEL_NUM1];
    size_t actual_tile_size = tile_size + kernelSize1 -1;

    memset(tile, 0, sizeof(tile));
    memset(intermediate_val, 0, sizeof(intermediate_val));
    memset(intermediate2_val, 0, sizeof(intermediate2_val));
    
    for (int i = 0; i < CHANNEL_NUM1; i++) {
        intermediate[i] = &intermediate_val[i][0];
        intermediate2[i] = &intermediate2_val[i][0];
    }
    
    // real_t output_tile[channel_number2];
    // for (int i = 0; i< channel_number2;i++){
    //     output_tile[i] = 0.0f;
    // }

    // int outputSize = (input_size -kernelSize1 +1)/2 - kernelSize2 + 1;
    // printf("outputSize: %d \n", outputSize);
    for (int i = 0; i < input_size - kernelSize2 ; i+= tile_size){

        fill_tile(tile, input_data, i, actual_tile_size); //comment for testing
        // fill_tile(tile, input_data, i, tile_size); // for test only

        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1, actual_tile_size, kernelSize1);
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        multi_channel_aggregation_and_pooling(intermediate2, output_tile, kernel2, channel_number1, channel_number2, tile_size/2, kernelSize2,i/2,(input_size -kernelSize1 +1)/2);
    }

    // print_array_output_tile(output_tile, channel_number2);
    // for(int i = 0; i< channel_number2;i++){
    //     output_tile[i] /= outputSize;;
    //     // Since the bias is the same for every element of the same channel
    //     // It is added outputSize times to a channel, so we just have to add it once after division
    //     output_tile[i] += convbias2[i];
    // }
    // print_array_output_tile(output_tile, channel_number2);

    // mlp(output_tile, output, channel_number2, 64, 2, weight1, weight2, fcbias1, fcbias2 );
}


// static real_t input_data[16000];
static float ring_buffer[16000];
int main(void){
    // Test
    
    int channel_number1 = CHANNEL_NUM1;
    int kernelSize1 = KERNEL_SIZE1;
    int channel_number2 = CHANNEL_NUM2;
    int kernelSize2 = KERNEL_SIZE2;
    int tile_size = TILE_SIZE;
    int input_size = INPUT_SIZE;
    real_t output[2];
    int outputSize = (48000 -kernelSize1 +1)/2 - kernelSize2 + 1;
    
    
    // set microphone variables
    int sample = 0;

    puts("This test will sample all available ADC lines once every 62ms with\n"
         "a 10-bit resolution and print the sampled results to STDIO\n\n");

    int result;

    result = gpio_init_out_highdrive(RUN_MIC_PIN);

    if (result == 0) {
        printf("Success!\n");
    }
    else {
        printf("Failure!\n");
    }
    gpio_set(RUN_MIC_PIN);

    /* initialize all available ADC lines */
    if (adc_init(ADC_LINE(3)) < 0) {
            printf("Initialization of ADC_LINE(%u) failed\n", 3);
            return 1;
        } else {
            printf("Successfully initialized ADC_LINE(%u)\n", 3);
        }
    
    real_t output_tile[channel_number2];
    for (int k = 0; k< channel_number2;k++){
        output_tile[k] = 0.0f;
    }

    unsigned int i = 0;
    unsigned int j = 0;
    while (1) {
            const int BIAS_10_BITS = 398;

            // Read data from ADC
            sample = adc_sample(ADC_LINE(3), RES) - BIAS_10_BITS;
            // printf("ADC_LINE(%u): %i\n", 3, sample);


            // Save in buffer normalized values from ADC
            ring_buffer[i] = sample / 1023.0f;
            i++;
            
            
            // If 16000 samples are written to buffer we give buffer to model as for partial convolution
            if (i == sizeof(ring_buffer) / sizeof(float)) {
                printf("Buffer full! Triggering inference...\n");
                print_buffer_details(ring_buffer, sizeof(ring_buffer) / sizeof(float));
                i = 0;

                // Model works with 3 sec audio but buffer has only 1 sec signal. So we use partial convolution - we aggregate output_tile
                CNN_model_inference((real_t*)ring_buffer, output, conv1weight, channel_number1, kernelSize1, conv2weight, channel_number2, kernelSize2, tile_size, input_size, fc1weight, fc2weight,fc1bias,fc2bias, conv1bias, conv2bias, output_tile);
                print_array_output_tile(output_tile, channel_number2);

                j++;
                printf("Debug: j = %d\n", j);
            }

            // If output_tile has data from 3 seconds (3 buffers) do a prediction
            if (j == 3) {
                printf("outputSize: %d \n", outputSize);
                for(int l = 0; l< channel_number2;l++){
                    output_tile[l] /= outputSize;;
                    output_tile[l] += conv2bias[l];
                }

                mlp(output_tile, output, channel_number2, 64, 2, fc1weight, fc2weight,fc1bias,fc2bias);

                printf("Inference output: \n");
                print_array(output,2);

                j = 0;
                for (int k = 0; k< channel_number2;k++){
                    output_tile[k] = 0.0f;
                }

            } 

            ztimer_sleep(ZTIMER_USEC, DELAY_US);
            

    }

    return 0; 
}
