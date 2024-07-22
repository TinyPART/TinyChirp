#ifndef SOS_FILTER_H
#define SOS_FILTER_H
#include "global.h"

#define SOS_ORDER 2

typedef float sos_coeff_t;
typedef io_t sos_io_t;

// Structure to hold the coefficients and states for one SOS section
typedef struct {
    sos_coeff_t b[SOS_ORDER + 1];  // Numerator coefficients
    sos_coeff_t a[SOS_ORDER + 1];      // Denominator coefficients (a0 is assumed to be 1 and not stored)
    sos_io_t w[SOS_ORDER];      // Delay line (filter states)
} SOS;

// Function to initialize the SOS filter structure
void init_sos(SOS* sos, const sos_coeff_t* b_coeffs, const sos_coeff_t* a_coeffs);

// Function to apply the SOS filter to an input sample
sos_io_t apply_sos(SOS* sos, sos_io_t input);

// Function to apply a cascaded SOS filter to a buffer of samples
void apply_cascaded_sos(SOS* sos_filters, int num_sections, const sos_io_t* input, sos_io_t* output, int length);

#endif // FIR_FILTER_H