import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    recall_score, 
    accuracy_score, 
    precision_score, 
    f1_score
)

def squeeze(audio, labels=None):
    """
    This dataset only contains single channel audio, so use
    the tf.squeeze function to drop the extra axis
    """
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def create_spectrogram_features(audio, desired_length, sample_rate):
    # Make all audios the same length 
    # audio, _ = squeeze(audio)
    audio_length = tf.shape(audio)[0]
    if audio_length < desired_length:
        audio = tf.pad(audio, [[0, desired_length - audio_length]], mode='CONSTANT')
    else:
        audio = audio[:desired_length]

    #Create log Mel spectrogram
    stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256,
                       fft_length=1024)
    spectrogram = tf.abs(stfts)
    # Warp the linear scale spectrogram into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrogram = tf.tensordot(
    spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    # Get back color channel axis to make features accetable for the model input shape
    log_mel_spectrogram_with_color_channel = tf.expand_dims(log_mel_spectrogram, axis=-1).numpy()
    return log_mel_spectrogram_with_color_channel

def evaluate_prediction(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'F1-score: {f1score * 100:.2f}%')

def lite_model_from_file_predicts_dataset(model_path, x_data, y_true, input_data_uint8_type=False):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pred_labels = []
    for i in range(len(x_data)):
        input_data = np.expand_dims(x_data[i], axis=0)
        if input_data_uint8_type == True:
            interpreter.set_tensor(input_details[0]["index"], tf.cast(input_data, tf.uint8))
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        pred_prob = interpreter.get_tensor(output_details[0]['index'])
        pred_label = np.argmax(pred_prob, axis=1)
        pred_labels.append(pred_label)

    evaluate_prediction(y_true, pred_labels)
    return pred_labels


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')
