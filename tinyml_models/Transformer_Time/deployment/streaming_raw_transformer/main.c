#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <inttypes.h>
#include "xtimer.h"
#include "conv_array_data.h"
#include "transformer_array_data.h"

#define MAX_LINE_LENGTH 1024
typedef struct{
    int in_features;
    int out_features;
    float * weights;
    float *bias;
} Linear;

typedef struct{
    int n_embd;
    int size;
    float *query;
    float *key;
    float *value;

} Attention;

typedef struct{
    Attention* head;
    int hidden_size;
    float* ln1;
    float* ln2;
    float* ln1bias;
    float* ln2bias;
    float* linear1;
    float* linear1bias;
    float* linear2;
    float* linear2bias;
} TransformerBlock;

Linear * create_linear(int in_features, int out_features, float* weights, float *bias ){
    Linear* res = (Linear *) malloc(sizeof(Linear));
    res->in_features = in_features;
    res->out_features = out_features;
    res->weights = weights;
    res->bias = bias;
    return res;
}

Attention * create_attention(int n_embd,int size, float *query, float* key, float *value){
    Attention* res = (Attention*)malloc(sizeof(Attention));
    res->n_embd = n_embd;
    res->size = size;
    res->query = query;
    res->key = key;
    res->value = value;
    return res;
}

TransformerBlock* create_block(Attention* head, int hidden_size, float* ln1, float* ln2, float* linear1, float* linear2, float* bias1, float* bias2, float* ln1bias, float* ln2bias){
    TransformerBlock* res = (TransformerBlock*)malloc(sizeof(TransformerBlock));
    res->hidden_size = hidden_size;
    res->head = head;
    res->ln1 = ln1;
    res->ln2 = ln2;
    res->ln1bias = ln1bias;
    res->ln2bias = ln2bias;
    res->linear1 = linear1;
    res->linear2 = linear2;
    res->linear1bias = bias1;
    res->linear2bias = bias2;
    return res;
}

void free_linear(Linear *layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer);
}

void free_attention(Attention *head) {
    free(head->query);
    free(head->key);
    free(head->value);
    free(head);
}

void print_array(float* array, int size){
    for (int i = 0;i<size;i++){
        printf("%f ",array[i]);
    }
    printf("\n");
}

void linear_forward(int in_features, int out_features, float* input, float* output, float* weights, float* bias ){
    for (int i = 0; i < out_features; i++){
        output[i] = 0.0;
        for (int j = 0; j < in_features; j++){
            output[i] += input[j] * weights[i*in_features +j];
        }
        if (bias){
            output[i] += bias[i];
        }
    }
}

float dot_product(float *a, float *b, int size){
    float sum = 0.0;
    for (int i = 0; i< size; i++){
        sum += a[i] * b[i];
    }
    return sum;
}

void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i] - max_val);
        // The substraction by the max val is used for numerical stability
        // DOesn't change the result because softmax(x) = softmax(x+c)
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}
void scaled_dot_product_attention(float *query, float *key, float *value, int size, int seq_length, float *output) {
    float *scores = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i+= size) {
        scores[i/size] = dot_product(query, &key[i], size) / sqrt((float)size);
    }
    float *attention_weights = (float *)malloc(seq_length * sizeof(float));
    softmax(scores, attention_weights, seq_length);
    for (int i = 0; i < size; ++i) {
        output[i] = 0.0;
        for (int j = 0; j < seq_length; ++j) {
            output[i] += attention_weights[j] * value[j * size + i];
        }
    }
    free(scores);
    free(attention_weights);
}

void add_aggregation(float* input, float* add, int size){
    for (int i = 0;i<size;i++){
        input[i] += add[i];
    }
}

void layer_norm(float* input, float* output, int size, float* gamma, float* beta, float epsilon) {
    float mean = 0.0f;
    float variance = 0.0f;
    float inv_stddev;

    // Calculate mean
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;

    // Calculate variance
    for (int i = 0; i < size; i++) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;

    // Calculate inverse standard deviation
    inv_stddev = 1.0f / sqrt(variance + epsilon);

    // Normalize and apply scale (gamma) and shift (beta)
    for (int i = 0; i < size; i++) {
        output[i] = gamma[i] * ((input[i] - mean) * inv_stddev) + beta[i];
    }
}
void inplace_relu(float* input, int size){
    for (int i = 0; i< size;i++){
        if (input[i] <=0){
            input[i] = 0;
        }
    }
    return;
}
typedef float real_t;


real_t **allocate_2d_array(int rows, int cols){
    real_t ** array = malloc(rows * sizeof(real_t *));
    for (int i = 0; i< rows; i++){
        array[i] = malloc(cols * sizeof(real_t));
        for (int j = 0;j<cols;j++){
            array[i][j] = 0.0f;
        }
    }
    return array;
}

void free_2d_array(real_t **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}


void conv1d_and_relu_multi_channel(real_t *input, real_t** kernel, real_t ** output, real_t* bias , int channel_number, int inputSize, int kernelSize){
    for (int channel = 0; channel < channel_number; channel++){
        for(int i = 0; i< inputSize - kernelSize + 1; i++){
            output[channel][i] = 0.0f;
            for(int j = 0; j< kernelSize; j++){
                output[channel][i] += input[i + j] * kernel[channel][j];
            }
            output[channel][i] += bias[channel];
            if (output[channel][i] <= 0){
                output[channel][i] = 0.0f;
            }
        }
    }
}



void fill_tile(real_t* tile, real_t* input_data, int i ,int tile_size){
    for (int j = 0; j < tile_size; j++ ){
        tile[j] = input_data[i + j];
    }
}

void maxpool1d_channel(real_t** tile, int tile_size, int channel_number, real_t** output_tile){
    for (int channel = 0; channel < channel_number; channel++){
        for (int i= 0; i< tile_size; i+=2){
            if (tile[channel][i]> tile[channel][i+1] ){
                output_tile[channel][i/2] = tile[channel][i];
                
            }
            else{
                output_tile[channel][i/2] = tile[channel][i+1];
            }
        }
    }
}
void transformer_forward(Linear * layer, TransformerBlock * block, float* input,float* ln_f, float* ln_fbias, float* output, float* projweight, float* projbias ){
    float* query = (float*) malloc(block->head->size * sizeof(float));
    float* key = (float*) malloc(block->head->size * sizeof(float));
    float* value = (float*) malloc(block->head->size * sizeof(float));
    float* normalized = (float*)malloc(block->head->size * sizeof(float));
    layer_norm(input,normalized, block->head->size, block->ln1, block->ln1bias, 1e-5 );

    linear_forward(block->head->n_embd, block->head->size, normalized, query, block->head->query,NULL);
    linear_forward(block->head->n_embd, block->head->size, normalized, key, block->head->key,NULL);
    linear_forward(block->head->n_embd, block->head->size, normalized, value, block->head->value,NULL);
    //Check
    
    float* attention_output = (float*)malloc(block->head->size * sizeof(float));
    //Self attention layer norm
    //self.sa(self.ln1(x))
    scaled_dot_product_attention(query, key, value, block->head->size,1, attention_output);
    
    //proj
    float * projected = (float*)malloc(block->head->size * sizeof(float));
    linear_forward(block->head->size, block->head->size, attention_output,projected,projweight,projbias );
    //check
    // x = x + self.sa(ln1(x))
    add_aggregation(projected,input,block->head->size);
    //check
    //x = x + ffwd(ln2(x))
    layer_norm(projected, normalized, block->head->size,block->ln2, block->ln2bias,1e-5);
    float* hidden_layer = (float*)malloc(block->hidden_size * sizeof(float));
    linear_forward(block->head->size,block->hidden_size,normalized, hidden_layer,block->linear1, block->linear1bias);
    inplace_relu(hidden_layer, block->hidden_size);
    linear_forward(block->hidden_size,block->head->size,hidden_layer, normalized,block->linear2, block->linear2bias);
    //check
    add_aggregation(normalized, projected, block->head->size);
    //x  = ln_f(x)
    layer_norm(normalized, attention_output, block->head->size,ln_f,ln_fbias,1e-5);
    //self.head(x)
    linear_forward(layer->in_features, layer->out_features, attention_output, output, layer->weights, layer->bias);
}
void adpool_cumulative(real_t* output, real_t** multi_channel_tile, int tile_size, int output_size){
    for (int i = 0;i<output_size;i++){
        for( int j = 0;j<tile_size;j++){
            output[i] += multi_channel_tile[i][j];
        }
    }
    return;
}
void Raw_audio_model_inference(real_t* input_data, real_t* output ,real_t** kernel1, int channel_number1, int kernelSize1, int tile_size, int input_size, real_t* convbias1, TransformerBlock *block, Linear* layer, real_t* ln_fweight, real_t* projweight, real_t* projbias){
//    real_t tile[tile_size + kernelSize1 -1]; // take too much stack!
    real_t* tile = malloc(sizeof(real_t) * (tile_size + kernelSize1 -1));
    real_t**  intermediate = allocate_2d_array(channel_number1, tile_size);
    real_t** intermediate2 = allocate_2d_array(channel_number1, tile_size/2);
    real_t pooled_result[block->head->n_embd];
    for (int i = 0; i< block->head->n_embd;i++){
        pooled_result[i] = 0.0f;
    }

    int outputSize = (input_size -kernelSize1 +1)/2;
    for (int i = 0; i <= input_size - kernelSize1; i+= tile_size){
        fill_tile(tile, input_data, i, tile_size + kernelSize1 -1);
        conv1d_and_relu_multi_channel(tile, kernel1, intermediate, convbias1,channel_number1,tile_size + kernelSize1 - 1, kernelSize1);
        maxpool1d_channel(intermediate, tile_size,channel_number1,intermediate2);
        adpool_cumulative(pooled_result, intermediate2, tile_size/2, channel_number1);
    }
    for (int i = 0;i<channel_number1;i++){
        pooled_result[i] /= outputSize;
    }
    transformer_forward(layer, block, pooled_result, ln_fweight,ln_fbias, output, projweight,projbias);
    
}


static real_t input_data[16000];
int main(void){
    // Test
    
    puts("Début des hostilités");
    
    int channel_number1 = 16;
    int kernelSize1 = 3;
    int tile_size = 64;
    int input_size = 16000;
    int n_embd = 16;
    int block_size= 16;
    int num_classes = 2;
    int hidden_size = 32;

    for (int i = 0; i < 16000;i++){
        input_data[i] = i/160.0f;
    }

    Linear * layer = create_linear(block_size, num_classes,weight,bias);
    Attention * head = create_attention(n_embd,block_size,queryweight,keyweight,valueweight);
    TransformerBlock* block = create_block(head, hidden_size, ln1weight,ln2weight,ffwd0weight,ffwd2weight,ffwd0bias,ffwd2bias,ln1bias,ln2bias);
    
    real_t output[2];

    puts("Ok on lance l'inference");
    
    uint32_t inference_duration;
    inference_duration = xtimer_now_usec();
    Raw_audio_model_inference(input_data,output,conv1weight,channel_number1,kernelSize1,tile_size,input_size,conv1bias,block,layer,ln_fweight,projweight,projbias);
    inference_duration = xtimer_now_usec() - inference_duration;
    
    
    
    printf("\nLe résultat est : \n");
    print_array(output,2);
    printf("inference duration in usec: %" PRIu32 " \n", inference_duration);
    

}
