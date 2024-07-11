#include "arm_math.h"
#include "arm_const_structs.h"
#include <math.h>
#include "xtimer.h"

#define SAMPLE_RATE        16000
#define INPUT_LEN_SEC      1
#define FRAME_LENGTH       1024
#define HOP_LENGTH         256
#define NUM_FRAMES         184
#define NUM_BINS           513
#define MEL_BINS           80
#define FFT_SIZE           1024

#define MEL_LOW_FREQ       80.0f
#define MEL_HIGH_FREQ      8000.0f

#include "coeff.h"
//float32_t input_audio[(NUM_FRAMES-1) * HOP_LENGTH + FRAME_LENGTH];
static float32_t input_audio[SAMPLE_RATE * INPUT_LEN_SEC];

// static float32_t fft_output[NUM_FRAMES * NUM_BINS];
static float32_t mel_spectrogram[NUM_FRAMES * MEL_BINS];

// Function prototypes
void apply_hann_window(float32_t *input, const float32_t *window, uint32_t length);
void compute_stft(float32_t *input, float32_t *output, uint32_t frame_length, uint32_t hop_length);
void compute_mel_spectrogram(float32_t *stft_output, float32_t *mel_output, uint32_t num_frames, uint32_t num_bins, uint32_t mel_bins);
void compute_log_mel_spectrogram(float32_t *mel_output, uint32_t num_frames, uint32_t mel_bins);
void compute_stft_mel(float32_t *input, float32_t *mel_output, uint32_t frame_length, uint32_t hop_length, uint32_t num_frames, uint32_t num_bins, uint32_t mel_bins);
// Main function
int main(void) {
    // Initialize Hann window
//    for (uint32_t i = 0; i < FRAME_LENGTH; i++) {
//        hann_window[i] = 0.5f * (1.0f - arm_cos_f32((2.0f * PI * i) / (FRAME_LENGTH - 1)));
//    }

   for (uint32_t i = 0; i < sizeof(input_audio) / sizeof(float32_t); i++) {
       input_audio[i] = (sizeof(input_audio) / sizeof(float32_t) - i) / (float)(sizeof(input_audio) / sizeof(float32_t));
   }

    uint32_t start, end;
    start = xtimer_now_usec();

    // Apply Hann window and compute STFT
    // compute_stft(input_audio, fft_output, FRAME_LENGTH, HOP_LENGTH);

    // Compute Mel spectrogram
    // compute_mel_spectrogram(fft_output, mel_spectrogram, NUM_FRAMES, NUM_BINS, MEL_BINS);

    //Merge STFT and Mel into one
    compute_stft_mel(input_audio, mel_spectrogram, FRAME_LENGTH, HOP_LENGTH, NUM_FRAMES, NUM_BINS, MEL_BINS);

    // Compute log Mel spectrogram
    compute_log_mel_spectrogram(mel_spectrogram, NUM_FRAMES, MEL_BINS);

   end = xtimer_now_usec();
   printf("Preprocess Latency usec: %ld\n", (long int)(end - start));

    // Now mel_spectrogram contains the log Mel-Spectrogram

    return 0;
}

void apply_hann_window(float32_t *input, const float32_t *window, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        input[i] *= window[i];
    }
}

static float32_t buffer[FRAME_LENGTH];
static float32_t output_buffer[FRAME_LENGTH];

void compute_stft_mel(float32_t *input, float32_t *mel_output, uint32_t frame_length, uint32_t hop_length, uint32_t num_frames, uint32_t num_bins, uint32_t mel_bins) {

    arm_rfft_fast_instance_f32 fft_instance;

    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

    float32_t *stft_output = output_buffer;

    for (uint32_t i = 0; i < num_frames; i++) {
        
        memcpy(buffer, &input[(i * hop_length) % (SAMPLE_RATE * INPUT_LEN_SEC - FRAME_LENGTH)], frame_length * sizeof(float32_t));
        apply_hann_window(buffer, hann_window, frame_length);

        // Compute FFT
        arm_rfft_fast_f32(&fft_instance, buffer, output_buffer, 0);
        
        for (uint32_t j = 0; j < mel_bins; j++) {
            mel_output[i * mel_bins + j] = 0.0f;
            for (uint32_t k = 0; k < num_bins; k++) {
                mel_output[i * mel_bins + j] += stft_output[k] * mel_filterbank[j * num_bins + k];
            }
        }
    }
}

void compute_stft(float32_t *input, float32_t *output, uint32_t frame_length, uint32_t hop_length) {
    arm_rfft_fast_instance_f32 fft_instance;

    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

    for (uint32_t i = 0; i < NUM_FRAMES; i++) {
        // Copy and window the frame
//        memcpy(buffer, &input[i * hop_length], frame_length * sizeof(float32_t));
        //var 1 circulate for simulation
        memcpy(buffer, &input[(i * hop_length) % (SAMPLE_RATE * INPUT_LEN_SEC - FRAME_LENGTH)], frame_length * sizeof(float32_t));
        apply_hann_window(buffer, hann_window, frame_length);

        // Compute FFT
        arm_rfft_fast_f32(&fft_instance, buffer, output_buffer, 0);
        
        memcpy(output + (i * NUM_BINS), output_buffer, NUM_BINS * sizeof(float32_t));

    }
}

void compute_mel_spectrogram(float32_t *stft_output, float32_t *mel_output, uint32_t num_frames, uint32_t num_bins, uint32_t mel_bins) {
    for (uint32_t i = 0; i < num_frames; i++) {
        for (uint32_t j = 0; j < mel_bins; j++) {
            mel_output[i * mel_bins + j] = 0.0f;
            for (uint32_t k = 0; k < num_bins; k++) {
                mel_output[i * mel_bins + j] += stft_output[i * num_bins + k] * mel_filterbank[j * num_bins + k];
            }
        }
    }
}

void compute_log_mel_spectrogram(float32_t *mel_output, uint32_t num_frames, uint32_t mel_bins) {
    for (uint32_t i = 0; i < num_frames * mel_bins; i++) {
        mel_output[i] = logf(mel_output[i] + 1e-6f); // Adding a small value to avoid log(0)
    }
}

