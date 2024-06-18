/*
 * Copyright (C) 2023 Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     apps
 * @{
 *
 * @file
 * @brief       Transformer Audio Application
 *
 * @author      Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * @}
 */

#include <stdio.h>
#include <string.h>
#include "xtimer.h"
#include "random.h"
#include "stdio_base.h"
#include "mlmci.h"

#include <tvmgen_default.h>
extern mlmodel_t *model_ptr;

void model_inference(void)
{       


        //set input data
        mlmodel_iovar_t *input = mlmodel_get_input_variable(model_ptr, 0);
        float* input_val = input->values; //pointer to input data

        
        //run inference
        int start, end;
        start =  xtimer_now_usec();
        int ret_val = mlmodel_inference(model_ptr);
        end =  xtimer_now_usec();
        printf("inference usec: %ld, ret: %d \n", (long int)(end - start), ret_val);
        
        //fetch model output
        mlmodel_iovar_t *output = mlmodel_get_output_variable(model_ptr, 0);
        float* output_val = output->values; //pointer to output data
}


int main(void)
{
    mlmodel_init(model_ptr);
    mlmodel_set_global_model(model_ptr);

    model_inference();

    return 0;
}
