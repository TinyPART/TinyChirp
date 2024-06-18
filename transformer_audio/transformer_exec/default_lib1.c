// tvm target: c -keys=arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840
#define TVM_EXPORTS
//#include "tvm/runtime/c_runtime_api.h"
//#include "tvm/runtime/c_backend_api.h"
#define TVM_DLL
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_workspace_1_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean_1(float* p0, float* T_divide, uint8_t* global_workspace_17_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean_2(float* p0, float* T_divide, uint8_t* global_workspace_26_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_5_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_1(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_7_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_2(float* p0, float* p1, float* T_batch_matmul_NT, uint8_t* global_workspace_8_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_3(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_12_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_4(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_15_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_5(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_21_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_6(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_24_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_7(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_30_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_broadcast_to(float* p0, float* p1, float* T_broadcast_to, uint8_t* global_workspace_13_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax(float* p0, float* T_softmax_norm, uint8_t* global_workspace_10_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add(float* p0, float* p1, float* T_add, uint8_t* global_workspace_31_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_add(float* p0, float* p1, float* p2, float* T_add, uint8_t* global_workspace_16_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_add_1(float* p0, float* p1, float* p2, float* T_add, uint8_t* global_workspace_25_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_nn_relu_broadcast_to_reshape(float* p0, float* p1, float* T_reshape, uint8_t* global_workspace_22_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_squeeze_multiply(float* p0, float* T_multiply, uint8_t* global_workspace_9_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_3_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_1(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_19_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_2(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_28_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape(float* p0, float* T_reshape, uint8_t* global_workspace_4_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_1(float* p0, float* T_reshape, uint8_t* global_workspace_6_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_2(float* p0, float* T_reshape, uint8_t* global_workspace_11_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_3(float* p0, float* T_reshape, uint8_t* global_workspace_14_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_4(float* p0, float* T_reshape, uint8_t* global_workspace_20_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_5(float* p0, float* T_reshape, uint8_t* global_workspace_23_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_6(float* p0, float* T_reshape, uint8_t* global_workspace_29_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_2_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance_1(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_18_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance_2(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_27_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_buffer_var, float* _0_0_ln1_weight_buffer_var, float* _0_0_ln1_bias_buffer_var, float* _0_0_sa_head_query_weight_buffer_var, float* _0_0_sa_head_key_weight_buffer_var, float* _0_0_sa_head_value_weight_buffer_var, float* _0_0_sa_proj_weight_buffer_var, float* _0_0_sa_proj_bias_buffer_var, float* _0_0_ln2_weight_buffer_var, float* _0_0_ln2_bias_buffer_var, float* _0_0_ffwd_0_weight_buffer_var, float* _0_0_ffwd_0_bias_buffer_var, float* _0_0_ffwd_2_weight_buffer_var, float* _0_0_ffwd_2_bias_buffer_var, float* _1_weight_buffer_var, float* _1_bias_buffer_var, float* _2_weight_buffer_var, float* _2_bias_buffer_var, float* output_buffer_var, uint8_t* global_workspace_0_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float expf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float sqrtf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float sqrtf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL float sqrtf(float);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean(float* p0, float* T_divide, uint8_t* global_workspace_1_var) {
  void* p0_red_let = (&(global_workspace_1_var[4]));
  ((float*)p0_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)p0_red_let)[0] = (((float*)p0_red_let)[0] + p0[k2]);
  }
  T_divide[0] = (((float*)p0_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean_1(float* p0, float* T_divide, uint8_t* global_workspace_17_var) {
  void* p0_red_let = (&(global_workspace_17_var[4]));
  ((float*)p0_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)p0_red_let)[0] = (((float*)p0_red_let)[0] + p0[k2]);
  }
  T_divide[0] = (((float*)p0_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_mean_2(float* p0, float* T_divide, uint8_t* global_workspace_26_var) {
  void* p0_red_let = (&(global_workspace_26_var[68]));
  ((float*)p0_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)p0_red_let)[0] = (((float*)p0_red_let)[0] + p0[k2]);
  }
  T_divide[0] = (((float*)p0_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_5_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 2; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_5_var[1024]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_5_var[1664]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_10 = (k_inner * 8);
      int32_t cse_var_9 = (cse_var_10 + 7);
      int32_t cse_var_8 = (cse_var_10 + 6);
      int32_t cse_var_7 = (cse_var_10 + 5);
      int32_t cse_var_6 = (cse_var_10 + 4);
      int32_t cse_var_5 = (cse_var_10 + 3);
      int32_t cse_var_4 = (cse_var_10 + 2);
      int32_t cse_var_3 = (cse_var_10 + 1);
      int32_t cse_var_2 = ((k_inner * 16) + cse_var_1);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] + (p0[k_inner] * p1[cse_var_2]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(cse_var_2 + 1)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(cse_var_2 + 2)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(cse_var_2 + 3)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(cse_var_2 + 4)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(cse_var_2 + 5)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(cse_var_2 + 6)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(cse_var_2 + 7)]));
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_1(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_7_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 2; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_7_var[1024]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_7_var[1728]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_10 = (k_inner * 8);
      int32_t cse_var_9 = (cse_var_10 + 7);
      int32_t cse_var_8 = (cse_var_10 + 6);
      int32_t cse_var_7 = (cse_var_10 + 5);
      int32_t cse_var_6 = (cse_var_10 + 4);
      int32_t cse_var_5 = (cse_var_10 + 3);
      int32_t cse_var_4 = (cse_var_10 + 2);
      int32_t cse_var_3 = (cse_var_10 + 1);
      int32_t cse_var_2 = ((k_inner * 16) + cse_var_1);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] + (p0[k_inner] * p1[cse_var_2]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(cse_var_2 + 1)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(cse_var_2 + 2)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(cse_var_2 + 3)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(cse_var_2 + 4)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(cse_var_2 + 5)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(cse_var_2 + 6)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(cse_var_2 + 7)]));
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_2(float* p0, float* p1, float* T_batch_matmul_NT, uint8_t* global_workspace_8_var) {
  void* T_batch_matmul_NT_global_rf_let = (&(global_workspace_8_var[0]));
  void* T_batch_matmul_NT_global_let = (&(global_workspace_8_var[64]));
  for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
    ((float*)T_batch_matmul_NT_global_rf_let)[k_inner] = 0.000000e+00f;
    ((float*)T_batch_matmul_NT_global_rf_let)[k_inner] = (((float*)T_batch_matmul_NT_global_rf_let)[k_inner] + (p0[k_inner] * p1[k_inner]));
  }
  ((float*)T_batch_matmul_NT_global_let)[0] = 0.000000e+00f;
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[0]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[1]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[2]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[3]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[4]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[5]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[6]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[7]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[8]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[9]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[10]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[11]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[12]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[13]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[14]);
  ((float*)T_batch_matmul_NT_global_let)[0] = (((float*)T_batch_matmul_NT_global_let)[0] + ((float*)T_batch_matmul_NT_global_rf_let)[15]);
  T_batch_matmul_NT[0] = ((float*)T_batch_matmul_NT_global_let)[0];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_3(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_12_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 2; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_12_var[1024]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_12_var[1664]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_10 = (k_inner * 8);
      int32_t cse_var_9 = (cse_var_10 + 7);
      int32_t cse_var_8 = (cse_var_10 + 6);
      int32_t cse_var_7 = (cse_var_10 + 5);
      int32_t cse_var_6 = (cse_var_10 + 4);
      int32_t cse_var_5 = (cse_var_10 + 3);
      int32_t cse_var_4 = (cse_var_10 + 2);
      int32_t cse_var_3 = (cse_var_10 + 1);
      int32_t cse_var_2 = ((k_inner * 16) + cse_var_1);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] + (p0[k_inner] * p1[cse_var_2]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(cse_var_2 + 1)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(cse_var_2 + 2)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(cse_var_2 + 3)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(cse_var_2 + 4)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(cse_var_2 + 5)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(cse_var_2 + 6)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(cse_var_2 + 7)]));
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_4(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_15_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 2; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_15_var[1024]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_15_var[1664]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_10 = (k_inner * 8);
      int32_t cse_var_9 = (cse_var_10 + 7);
      int32_t cse_var_8 = (cse_var_10 + 6);
      int32_t cse_var_7 = (cse_var_10 + 5);
      int32_t cse_var_6 = (cse_var_10 + 4);
      int32_t cse_var_5 = (cse_var_10 + 3);
      int32_t cse_var_4 = (cse_var_10 + 2);
      int32_t cse_var_3 = (cse_var_10 + 1);
      int32_t cse_var_2 = ((k_inner * 16) + cse_var_1);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] + (p0[k_inner] * p1[cse_var_2]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(cse_var_2 + 1)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(cse_var_2 + 2)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(cse_var_2 + 3)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(cse_var_2 + 4)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(cse_var_2 + 5)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(cse_var_2 + 6)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(cse_var_2 + 7)]));
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_5(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_21_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 4; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_21_var[2048]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_21_var[2816]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_10 = (k_inner * 8);
      int32_t cse_var_9 = (cse_var_10 + 7);
      int32_t cse_var_8 = (cse_var_10 + 6);
      int32_t cse_var_7 = (cse_var_10 + 5);
      int32_t cse_var_6 = (cse_var_10 + 4);
      int32_t cse_var_5 = (cse_var_10 + 3);
      int32_t cse_var_4 = (cse_var_10 + 2);
      int32_t cse_var_3 = (cse_var_10 + 1);
      int32_t cse_var_2 = ((k_inner * 32) + cse_var_1);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_10] + (p0[k_inner] * p1[cse_var_2]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(cse_var_2 + 1)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(cse_var_2 + 2)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(cse_var_2 + 3)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(cse_var_2 + 4)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(cse_var_2 + 5)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(cse_var_2 + 6)]));
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(cse_var_2 + 7)]));
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_6(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_24_var) {
  for (int32_t b_i_outer_fused_j_outer_fused = 0; b_i_outer_fused_j_outer_fused < 2; ++b_i_outer_fused_j_outer_fused) {
    int32_t cse_var_1 = (b_i_outer_fused_j_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_24_var[2176]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_24_var[2816]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      for (int32_t i_c_j_c_fused = 0; i_c_j_c_fused < 8; ++i_c_j_c_fused) {
        int32_t cse_var_3 = ((k_inner * 8) + i_c_j_c_fused);
        int32_t cse_var_2 = (((k_inner * 16) + cse_var_1) + i_c_j_c_fused);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = 0.000000e+00f;
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[cse_var_2]));
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[(k_inner + 16)] * p1[(cse_var_2 + 256)]));
      }
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_batch_matmul_NN[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_batch_matmul_NN[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_batch_matmul_NN[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_batch_matmul_NN[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_batch_matmul_NN[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_batch_matmul_NN[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_batch_matmul_NN[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_batch_matmul_NN[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_7(float* p0, float* p1, float* T_batch_matmul_NN, uint8_t* global_workspace_30_var) {
  void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_30_var[0]));
  void* T_batch_matmul_NN_global_let = (&(global_workspace_30_var[128]));
  for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
    int32_t cse_var_2 = (k_inner * 2);
    int32_t cse_var_1 = (cse_var_2 + 1);
    ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] = 0.000000e+00f;
    ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] + (p0[k_inner] * p1[cse_var_2]));
    ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_1] = 0.000000e+00f;
    ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_1] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_1] + (p0[k_inner] * p1[cse_var_1]));
  }
  for (int32_t ax2 = 0; ax2 < 2; ++ax2) {
    ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 2)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 4)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 6)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 10)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 12)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 14)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 18)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 20)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 22)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 26)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 28)]);
    ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 30)]);
  }
  T_batch_matmul_NN[0] = ((float*)T_batch_matmul_NN_global_let)[0];
  T_batch_matmul_NN[1] = ((float*)T_batch_matmul_NN_global_let)[1];
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_batch_matmul_broadcast_to(float* p0, float* p1, float* T_broadcast_to, uint8_t* global_workspace_13_var) {
  for (int32_t ax0_ax1_outer_fused_ax2_outer_fused = 0; ax0_ax1_outer_fused_ax2_outer_fused < 2; ++ax0_ax1_outer_fused_ax2_outer_fused) {
    int32_t cse_var_1 = (ax0_ax1_outer_fused_ax2_outer_fused * 8);
    void* T_batch_matmul_NN_global_rf_let = (&(global_workspace_13_var[0]));
    void* T_batch_matmul_NN_global_let = (&(global_workspace_13_var[512]));
    for (int32_t k_inner = 0; k_inner < 16; ++k_inner) {
      int32_t cse_var_2 = (k_inner * 8);
      ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] = 0.000000e+00f;
      if (k_inner < 1) {
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_2] + (p0[k_inner] * p1[((k_inner * 16) + cse_var_1)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 1)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_3 = (cse_var_2 + 1);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_3] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 1)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 2)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_4 = (cse_var_2 + 2);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_4] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 2)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 3)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_5 = (cse_var_2 + 3);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_5] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 3)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 4)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_6 = (cse_var_2 + 4);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_6] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 4)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 5)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_7 = (cse_var_2 + 5);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_7] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 5)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 6)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_8 = (cse_var_2 + 6);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_8] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 6)]));
      }
      ((float*)T_batch_matmul_NN_global_rf_let)[(cse_var_2 + 7)] = 0.000000e+00f;
      if (k_inner < 1) {
        int32_t cse_var_9 = (cse_var_2 + 7);
        ((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] = (((float*)T_batch_matmul_NN_global_rf_let)[cse_var_9] + (p0[k_inner] * p1[(((k_inner * 16) + cse_var_1) + 7)]));
      }
    }
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      ((float*)T_batch_matmul_NN_global_let)[ax2] = 0.000000e+00f;
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[ax2]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 8)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 16)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 24)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 32)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 40)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 48)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 56)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 64)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 72)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 80)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 88)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 96)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 104)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 112)]);
      ((float*)T_batch_matmul_NN_global_let)[ax2] = (((float*)T_batch_matmul_NN_global_let)[ax2] + ((float*)T_batch_matmul_NN_global_rf_let)[(ax2 + 120)]);
    }
    T_broadcast_to[cse_var_1] = ((float*)T_batch_matmul_NN_global_let)[0];
    T_broadcast_to[(cse_var_1 + 1)] = ((float*)T_batch_matmul_NN_global_let)[1];
    T_broadcast_to[(cse_var_1 + 2)] = ((float*)T_batch_matmul_NN_global_let)[2];
    T_broadcast_to[(cse_var_1 + 3)] = ((float*)T_batch_matmul_NN_global_let)[3];
    T_broadcast_to[(cse_var_1 + 4)] = ((float*)T_batch_matmul_NN_global_let)[4];
    T_broadcast_to[(cse_var_1 + 5)] = ((float*)T_batch_matmul_NN_global_let)[5];
    T_broadcast_to[(cse_var_1 + 6)] = ((float*)T_batch_matmul_NN_global_let)[6];
    T_broadcast_to[(cse_var_1 + 7)] = ((float*)T_batch_matmul_NN_global_let)[7];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax(float* p0, float* T_softmax_norm, uint8_t* global_workspace_10_var) {
  void* T_softmax_maxelem_let = (&(global_workspace_10_var[0]));
  void* T_softmax_expsum_let = (&(global_workspace_10_var[4]));
  ((float*)T_softmax_maxelem_let)[0] = -3.402823e+38f;
  float v_ = ((float*)T_softmax_maxelem_let)[0];
  float v__1 = p0[0];
  ((float*)T_softmax_maxelem_let)[0] = ((v_) > (v__1) ? (v_) : (v__1));
  ((float*)T_softmax_maxelem_let)[0] = expf((p0[0] - ((float*)T_softmax_maxelem_let)[0]));
  ((float*)T_softmax_expsum_let)[0] = 0.000000e+00f;
  ((float*)T_softmax_expsum_let)[0] = (((float*)T_softmax_expsum_let)[0] + ((float*)T_softmax_maxelem_let)[0]);
  T_softmax_norm[0] = (((float*)T_softmax_maxelem_let)[0] / ((float*)T_softmax_expsum_let)[0]);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add(float* p0, float* p1, float* T_add, uint8_t* global_workspace_31_var) {
  for (int32_t ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    T_add[ax2_inner] = (p0[ax2_inner] + p1[ax2_inner]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_add(float* p0, float* p1, float* p2, float* T_add, uint8_t* global_workspace_16_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      T_add[cse_var_1] = (p2[cse_var_1] + (p0[cse_var_1] + p1[cse_var_1]));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_add_1(float* p0, float* p1, float* p2, float* T_add, uint8_t* global_workspace_25_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      T_add[cse_var_1] = (p2[cse_var_1] + (p0[cse_var_1] + p1[cse_var_1]));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_squeeze_add_nn_relu_broadcast_to_reshape(float* p0, float* p1, float* T_reshape, uint8_t* global_workspace_22_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 8; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      float v_ = p0[cse_var_1] + p1[cse_var_1];
      T_reshape[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_squeeze_multiply(float* p0, float* T_multiply, uint8_t* global_workspace_9_var) {
  T_multiply[0] = (p0[0] * 2.500000e-01f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_3_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      T_reshape[cse_var_1] = ((((p0[cse_var_1] - p1[0]) * (1.000000e+00f / sqrtf((p2[0] + 1.000000e-05f)))) * p3[cse_var_1]) + p4[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_1(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_19_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      T_reshape[cse_var_1] = ((((p0[cse_var_1] - p1[0]) * (1.000000e+00f / sqrtf((p2[0] + 1.000000e-05f)))) * p3[cse_var_1]) + p4[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_2(float* p0, float* p1, float* p2, float* p3, float* p4, float* T_reshape, uint8_t* global_workspace_28_var) {
  for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
    for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
      int32_t cse_var_1 = ((ax2_outer * 4) + ax2_inner);
      T_reshape[cse_var_1] = ((((p0[cse_var_1] - p1[0]) * (1.000000e+00f / sqrtf((p2[0] + 1.000000e-05f)))) * p3[cse_var_1]) + p4[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape(float* p0, float* T_reshape, uint8_t* global_workspace_4_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 16) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 64) + (ax2_inner * 16)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_1(float* p0, float* T_reshape, uint8_t* global_workspace_6_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 16) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 64) + (ax2_inner * 16)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_2(float* p0, float* T_reshape, uint8_t* global_workspace_11_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 16) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 64) + (ax2_inner * 16)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_3(float* p0, float* T_reshape, uint8_t* global_workspace_14_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 16) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 64) + (ax2_inner * 16)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_4(float* p0, float* T_reshape, uint8_t* global_workspace_20_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 8; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 32) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 64) + (ax2_inner * 16)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_5(float* p0, float* T_reshape, uint8_t* global_workspace_23_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 32; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
        T_reshape[(((ax0_ax1_fused * 16) + (ax2_outer * 4)) + ax2_inner)] = p0[(((ax2_outer * 128) + (ax2_inner * 32)) + ax0_ax1_fused)];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_transpose_broadcast_to_reshape_6(float* p0, float* T_reshape, uint8_t* global_workspace_29_var) {
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 16; ++ax0_ax1_fused) {
    for (int32_t ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      T_reshape[((ax0_ax1_fused * 2) + ax2_inner)] = p0[((ax2_inner * 16) + ax0_ax1_fused)];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_2_var) {
  void* T_multiply_red_let = (&(global_workspace_2_var[8]));
  ((float*)T_multiply_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)T_multiply_red_let)[0] = (((float*)T_multiply_red_let)[0] + ((p0[k2] - p1[0]) * (p0[k2] - p1[0])));
  }
  T_divide[0] = (((float*)T_multiply_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance_1(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_18_var) {
  void* T_multiply_red_let = (&(global_workspace_18_var[8]));
  ((float*)T_multiply_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)T_multiply_red_let)[0] = (((float*)T_multiply_red_let)[0] + ((p0[k2] - p1[0]) * (p0[k2] - p1[0])));
  }
  T_divide[0] = (((float*)T_multiply_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_variance_2(float* p0, float* p1, float* T_divide, uint8_t* global_workspace_27_var) {
  void* T_multiply_red_let = (&(global_workspace_27_var[72]));
  ((float*)T_multiply_red_let)[0] = 0.000000e+00f;
  for (int32_t k2 = 0; k2 < 16; ++k2) {
    ((float*)T_multiply_red_let)[0] = (((float*)T_multiply_red_let)[0] + ((p0[k2] - p1[0]) * (p0[k2] - p1[0])));
  }
  T_divide[0] = (((float*)T_multiply_red_let)[0] * 6.250000e-02f);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_buffer_var, float* _0_0_ln1_weight_buffer_var, float* _0_0_ln1_bias_buffer_var, float* _0_0_sa_head_query_weight_buffer_var, float* _0_0_sa_head_key_weight_buffer_var, float* _0_0_sa_head_value_weight_buffer_var, float* _0_0_sa_proj_weight_buffer_var, float* _0_0_sa_proj_bias_buffer_var, float* _0_0_ln2_weight_buffer_var, float* _0_0_ln2_bias_buffer_var, float* _0_0_ffwd_0_weight_buffer_var, float* _0_0_ffwd_0_bias_buffer_var, float* _0_0_ffwd_2_weight_buffer_var, float* _0_0_ffwd_2_bias_buffer_var, float* _1_weight_buffer_var, float* _1_bias_buffer_var, float* _2_weight_buffer_var, float* _2_bias_buffer_var, float* output_buffer_var, uint8_t* global_workspace_0_var) {
  void* sid_29_let = (&(global_workspace_0_var[1600]));
  void* sid_28_let = (&(global_workspace_0_var[0]));
  void* sid_30_let = (&(global_workspace_0_var[1536]));
  void* sid_27_let = (&(global_workspace_0_var[1696]));
  void* sid_35_let = (&(global_workspace_0_var[4]));
  void* sid_23_let = (&(global_workspace_0_var[0]));
  void* sid_34_let = (&(global_workspace_0_var[0]));
  void* sid_37_let = (&(global_workspace_0_var[0]));
  void* sid_33_let = (&(global_workspace_0_var[2688]));
  void* sid_18_let = (&(global_workspace_0_var[0]));
  void* sid_26_let = (&(global_workspace_0_var[4]));
  void* sid_24_let = (&(global_workspace_0_var[1664]));
  void* sid_25_let = (&(global_workspace_0_var[0]));
  void* sid_32_let = (&(global_workspace_0_var[1600]));
  void* sid_19_let = (&(global_workspace_0_var[4]));
  void* sid_47_let = (&(global_workspace_0_var[0]));
  void* sid_20_let = (&(global_workspace_0_var[1536]));
  void* sid_21_let = (&(global_workspace_0_var[0]));
  void* sid_31_let = (&(global_workspace_0_var[0]));
  void* sid_22_let = (&(global_workspace_0_var[1600]));
  void* sid_36_let = (&(global_workspace_0_var[2752]));
  void* sid_38_let = (&(global_workspace_0_var[2560]));
  void* sid_40_let = (&(global_workspace_0_var[0]));
  void* sid_39_let = (&(global_workspace_0_var[2048]));
  void* sid_41_let = (&(global_workspace_0_var[2752]));
  void* sid_42_let = (&(global_workspace_0_var[0]));
  void* sid_43_let = (&(global_workspace_0_var[64]));
  void* sid_44_let = (&(global_workspace_0_var[68]));
  void* sid_45_let = (&(global_workspace_0_var[256]));
  void* sid_46_let = (&(global_workspace_0_var[128]));
  if (tvmgen_default_fused_mean(input_buffer_var, sid_18_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_variance(input_buffer_var, sid_18_let, sid_19_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape(input_buffer_var, sid_18_let, sid_19_let, _0_0_ln1_weight_buffer_var, _0_0_ln1_bias_buffer_var, sid_20_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape(_0_0_sa_head_query_weight_buffer_var, sid_21_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul(sid_20_let, sid_21_let, sid_22_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_1(_0_0_sa_head_key_weight_buffer_var, sid_23_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_1(sid_20_let, sid_23_let, sid_24_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_2(sid_22_let, sid_24_let, sid_25_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_squeeze_multiply(sid_25_let, sid_26_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax(sid_26_let, sid_27_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_2(_0_0_sa_head_value_weight_buffer_var, sid_28_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_3(sid_20_let, sid_28_let, sid_29_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_broadcast_to(sid_27_let, sid_29_let, sid_30_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_3(_0_0_sa_proj_weight_buffer_var, sid_31_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_4(sid_30_let, sid_31_let, sid_32_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_squeeze_add_add(sid_32_let, _0_0_sa_proj_bias_buffer_var, input_buffer_var, sid_33_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_mean_1(sid_33_let, sid_34_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_variance_1(sid_33_let, sid_34_let, sid_35_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_1(sid_33_let, sid_34_let, sid_35_let, _0_0_ln2_weight_buffer_var, _0_0_ln2_bias_buffer_var, sid_36_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_4(_0_0_ffwd_0_weight_buffer_var, sid_37_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_5(sid_36_let, sid_37_let, sid_38_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_squeeze_add_nn_relu_broadcast_to_reshape(sid_38_let, _0_0_ffwd_0_bias_buffer_var, sid_39_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_5(_0_0_ffwd_2_weight_buffer_var, sid_40_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_6(sid_39_let, sid_40_let, sid_41_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_squeeze_add_add_1(sid_41_let, _0_0_ffwd_2_bias_buffer_var, sid_33_let, sid_42_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_mean_2(sid_42_let, sid_43_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_variance_2(sid_42_let, sid_43_let, sid_44_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add_broadcast_to_reshape_2(sid_42_let, sid_43_let, sid_44_let, _1_weight_buffer_var, _1_bias_buffer_var, sid_45_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_transpose_broadcast_to_reshape_6(_2_weight_buffer_var, sid_46_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_batch_matmul_7(sid_45_let, sid_46_let, sid_47_let, global_workspace_0_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_squeeze_add(sid_47_let, _2_bias_buffer_var, output_buffer_var, global_workspace_0_var) != 0 ) return -1;
  return 0;
}

