#include "signal_energy.h"

// Function to calculate the energy of a signal
io_t calculate_energy(const io_t* signal, int length) {
    io_t energy = 0.0;
    for (int i = 0; i < length; ++i) {
        energy += signal[i] * signal[i];
    }
    return energy;
}