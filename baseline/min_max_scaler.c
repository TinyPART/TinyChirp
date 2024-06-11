#include "min_max_scaler.h"
#include <float.h>

void find_min_max(const io_t* array, int length, min_max_scaler_t* scaler) {
    for (int i = 0; i < length; ++i) {
        if (array[i] < scaler->min) {
            scaler->min = array[i];
        }
        if (array[i] > scaler->max) {
            scaler->max = array[i];
        }
    }
}

// Function to apply Min-Max scaling on a sliding window
void apply_min_max_scaler(const io_t* input, int input_length, io_t* output, min_max_scaler_t* scaler) {

    // Handle the last segment if it's smaller than the window size
    for (int i = 0; i < input_length; ++i) {
        output[i] = (input[i] - scaler->min) / (scaler->max - scaler->min);
    }
}