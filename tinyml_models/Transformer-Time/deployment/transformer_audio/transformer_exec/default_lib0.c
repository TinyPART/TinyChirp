//#include "tvm/runtime/c_runtime_api.h"
#include <stdint.h>
#define TVM_DLL
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((section(".bss.noinit.tvm"), aligned(4)))
static uint8_t global_workspace[2848];
#include <tvmgen_default.h>
TVM_DLL int32_t tvmgen_default___tvm_main__(void* input,void* _0_0_ln1_weight,void* _0_0_ln1_bias,void* _0_0_sa_head_query_weight,void* _0_0_sa_head_key_weight,void* _0_0_sa_head_value_weight,void* _0_0_sa_proj_weight,void* _0_0_sa_proj_bias,void* _0_0_ln2_weight,void* _0_0_ln2_bias,void* _0_0_ffwd_0_weight,void* _0_0_ffwd_0_bias,void* _0_0_ffwd_2_weight,void* _0_0_ffwd_2_bias,void* _1_weight,void* _1_bias,void* _2_weight,void* _2_bias,void* output0,uint8_t* global_workspace_0_var);
int32_t tvmgen_default_run(struct tvmgen_default_inputs* inputs,struct tvmgen_default_outputs* outputs) {return tvmgen_default___tvm_main__(inputs->input,inputs->_0_0_ln1_weight,inputs->_0_0_ln1_bias,inputs->_0_0_sa_head_query_weight,inputs->_0_0_sa_head_key_weight,inputs->_0_0_sa_head_value_weight,inputs->_0_0_sa_proj_weight,inputs->_0_0_sa_proj_bias,inputs->_0_0_ln2_weight,inputs->_0_0_ln2_bias,inputs->_0_0_ffwd_0_weight,inputs->_0_0_ffwd_0_bias,inputs->_0_0_ffwd_2_weight,inputs->_0_0_ffwd_2_bias,inputs->_1_weight,inputs->_1_bias,inputs->_2_weight,inputs->_2_bias,outputs->output,((uint8_t*)&global_workspace));
}
#ifdef __cplusplus
}
#endif
