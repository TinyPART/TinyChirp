#ifndef TVMGEN_DEFAULT_H_
#define TVMGEN_DEFAULT_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Input tensor _0_0_sa_proj_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_SA_PROJ_BIAS_SIZE 64
/*!
 * \brief Input tensor _0_0_ffwd_2_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_FFWD_2_BIAS_SIZE 64
/*!
 * \brief Input tensor _0_0_ffwd_0_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_FFWD_0_BIAS_SIZE 128
/*!
 * \brief Input tensor input size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_INPUT_SIZE 64
/*!
 * \brief Input tensor _2_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__2_WEIGHT_SIZE 128
/*!
 * \brief Input tensor _0_0_sa_proj_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_SA_PROJ_WEIGHT_SIZE 1024
/*!
 * \brief Input tensor _0_0_ffwd_0_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_FFWD_0_WEIGHT_SIZE 2048
/*!
 * \brief Input tensor _0_0_ln2_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_LN2_BIAS_SIZE 64
/*!
 * \brief Input tensor _0_0_ln2_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_LN2_WEIGHT_SIZE 64
/*!
 * \brief Input tensor _0_0_sa_head_query_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_SA_HEAD_QUERY_WEIGHT_SIZE 1024
/*!
 * \brief Input tensor _0_0_sa_head_key_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_SA_HEAD_KEY_WEIGHT_SIZE 1024
/*!
 * \brief Input tensor _1_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__1_BIAS_SIZE 64
/*!
 * \brief Input tensor _0_0_ln1_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_LN1_WEIGHT_SIZE 64
/*!
 * \brief Input tensor _0_0_ln1_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_LN1_BIAS_SIZE 64
/*!
 * \brief Input tensor _0_0_sa_head_value_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_SA_HEAD_VALUE_WEIGHT_SIZE 1024
/*!
 * \brief Input tensor _2_bias size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__2_BIAS_SIZE 8
/*!
 * \brief Input tensor _0_0_ffwd_2_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__0_0_FFWD_2_WEIGHT_SIZE 2048
/*!
 * \brief Input tensor _1_weight size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT__1_WEIGHT_SIZE 64
/*!
 * \brief Output tensor output size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_OUTPUT_SIZE 8
/*!
 * \brief Input tensor pointers for TVM module "default" 
 */
struct tvmgen_default_inputs {
  void* input;
  void* _0_0_ln1_weight;
  void* _0_0_ln1_bias;
  void* _0_0_sa_head_query_weight;
  void* _0_0_sa_head_key_weight;
  void* _0_0_sa_head_value_weight;
  void* _0_0_sa_proj_weight;
  void* _0_0_sa_proj_bias;
  void* _0_0_ln2_weight;
  void* _0_0_ln2_bias;
  void* _0_0_ffwd_0_weight;
  void* _0_0_ffwd_0_bias;
  void* _0_0_ffwd_2_weight;
  void* _0_0_ffwd_2_bias;
  void* _1_weight;
  void* _1_bias;
  void* _2_weight;
  void* _2_bias;
};

/*!
 * \brief Output tensor pointers for TVM module "default" 
 */
struct tvmgen_default_outputs {
  void* output;
};

/*!
 * \brief entrypoint function for TVM module "default"
 * \param inputs Input tensors for the module 
 * \param outputs Output tensors for the module 
 */
int32_t tvmgen_default_run(
  struct tvmgen_default_inputs* inputs,
  struct tvmgen_default_outputs* outputs
);
/*!
 * \brief Workspace size for TVM module "default" 
 */
#define TVMGEN_DEFAULT_WORKSPACE_SIZE 2848

#ifdef __cplusplus
}
#endif

#endif // TVMGEN_DEFAULT_H_
