import tensorflow as tf
import random
import numpy as np
tf.random.set_seed(3407)
np.random.seed(3407)
random.seed(3407)


def squeeze(audio, labels=None):
    """
    This dataset only contains single channel audio, so use
    the tf.squeeze function to drop the extra axis
    """
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def create_log_mel_sprectrogram(signals, sample_rate=16000):
    stfts = tf.signal.stft(signals, frame_length=1024, frame_step=256,
                       fft_length=1024)
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms

def create_spectrograms_from_audio_dataset(dataset: tf.data.Dataset, sample_rate = 16000):
    dataset_without_color_channel = dataset.map(squeeze, tf.data.AUTOTUNE)
    return dataset_without_color_channel.map(
      map_func=lambda audio,label: (create_log_mel_sprectrogram(audio, sample_rate), label),
      num_parallel_calls=tf.data.AUTOTUNE)
