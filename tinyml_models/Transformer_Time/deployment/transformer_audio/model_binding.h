/* U-TOE Generated File */ 
#ifndef MODEL_BINDING_H 
#define MODEL_BINDING_H 
static inline void model_bind_tvm_iovars(const mlmodel_t *model, struct tvmgen_default_inputs *inputs, struct tvmgen_default_outputs *outputs) { 
inputs->_2_bias = model->params[0].values; 
inputs->_2_weight = model->params[1].values; 
inputs->_1_bias = model->params[2].values; 
inputs->_1_weight = model->params[3].values; 
inputs->_0_0_ffwd_2_bias = model->params[4].values; 
inputs->_0_0_ffwd_2_weight = model->params[5].values; 
inputs->_0_0_ffwd_0_bias = model->params[6].values; 
inputs->_0_0_ffwd_0_weight = model->params[7].values; 
inputs->_0_0_ln2_bias = model->params[8].values; 
inputs->_0_0_ln2_weight = model->params[9].values; 
inputs->_0_0_sa_proj_bias = model->params[10].values; 
inputs->_0_0_sa_proj_weight = model->params[11].values; 
inputs->_0_0_sa_head_value_weight = model->params[12].values; 
inputs->_0_0_sa_head_query_weight = model->params[13].values; 
inputs->_0_0_sa_head_key_weight = model->params[14].values; 
inputs->_0_0_ln1_bias = model->params[15].values; 
inputs->_0_0_ln1_weight = model->params[16].values; 
inputs->input = model->input_vars[0].values; 
outputs->output = model->output_vars[0].values; 
}
#endif