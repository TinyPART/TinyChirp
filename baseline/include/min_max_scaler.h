#ifndef MIN_MAX_SCALER_H
#define MIN_MAX_SCALER_H
#include "global.h"
#include <float.h>

typedef struct min_max_scaler
{
    io_t min;
    io_t max;
} min_max_scaler_t;

inline void init_min_max_scaler(min_max_scaler_t* scaler) {
    scaler->min = FLT_MAX;
    scaler->max = -FLT_MAX;
}

void find_min_max(const io_t* array, int length, min_max_scaler_t* scaler);


// Function to apply Min-Max scaling on a sliding window
void apply_min_max_scaler(const io_t* input, int input_length, io_t* output, min_max_scaler_t* scaler);

#endif // MIN_MAX_SCALER_H