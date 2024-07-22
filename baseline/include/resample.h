#ifndef RESAMPLE_H
#define RESAMPLE_H

#include "global.h"
// Function to downsample a signal using zero-order hold
void downsample(const io_t* input, int input_length, io_t* output, int* output_length, int factor);

#endif