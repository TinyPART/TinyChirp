#include "sos_filter.h"
#include <string.h>

// Function to initialize the SOS filter structure
void init_sos(SOS* sos, const sos_coeff_t* b_coeffs, const sos_coeff_t* a_coeffs) {
    memcpy(sos->b, b_coeffs, sizeof(sos_coeff_t) * (SOS_ORDER + 1));
    memcpy(sos->a, a_coeffs, sizeof(sos_coeff_t) * (SOS_ORDER + 1));
    memset(sos->w, 0, sizeof(sos_io_t) * SOS_ORDER);  // Initialize delay line to zero
}

// Function to apply the SOS filter to an input sample
sos_io_t apply_sos(SOS* sos, sos_io_t input){
    // Compute the output using the direct form II transposed structure
    sos_io_t y = sos->b[0] * input + sos->w[0];

    // Update the delay line
    sos->w[0] = sos->b[1] * input - sos->a[1] * y + sos->w[1];
    sos->w[1] = sos->b[2] * input - sos->a[2] * y;
    

    return y;
}

// Function to apply a cascaded SOS filter to a buffer of samples
void apply_cascaded_sos(SOS* sos_filters, int num_sections, const sos_io_t* input, sos_io_t* output, int length) {
    for (int i = 0; i < length; ++i) {
        sos_io_t temp = input[i];
        for (int j = 0; j < num_sections; ++j) {
            temp = apply_sos(&sos_filters[j], temp);
        }
        output[i] = temp;
    }
}

