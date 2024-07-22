#include "resample.h"

// Function to downsample a signal using zero-order hold
void downsample(const io_t* input, int input_length, io_t* output, int* output_length, int factor) {
    if (factor <= 0) {
        *output_length = 0;
        return;
    }

    // Calculate the length of the downsampled signal
    *output_length = (input_length) / factor;

    // Select every 'factor'-th sample from the input signal
    for (int i = 0; i < *output_length; ++i) {
        output[i] = input[i * factor];
    }
}