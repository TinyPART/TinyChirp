/* U-TOE Generated File */ 
#include <stddef.h>
#include <stdint.h>
#include "mlmci.h"
__attribute__ ((aligned(4)))
static uint8_t input[64]; 
__attribute__ ((aligned(4)))
static uint8_t output[8]; 
static const mlmodel_iovar_t _model_input_vars[] = { 
{.name = "input",.values = (uint8_t*)input,.num_bytes = sizeof(input),},
};
static const mlmodel_iovar_t _model_output_vars[] = { 
{.name = "output",.values = (uint8_t*)output,.num_bytes = sizeof(output),},
};
const mlmodel_iovar_t* get_model_input_vars(void) {
return _model_input_vars;
}
const mlmodel_iovar_t* get_model_output_vars(void) {
return _model_output_vars;
}
