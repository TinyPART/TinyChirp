import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import time
import random
tf.random.set_seed(3407)
np.random.seed(3407)
random.seed(3407)



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

def plot_spectrogram(spectrogram, ax):
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, spectrogram)

def lite_model_predict_dataset(tf_lite_model, x_data, y_true, input_data_uint8_type=False):
    interpreter = tf.lite.Interpreter(model_content=tf_lite_model)
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

    accuracy = accuracy_score(y_true, pred_labels)
    recall = recall_score(y_true, pred_labels, average='macro')
    precision = precision_score(y_true, pred_labels)
    f1score = f1_score(y_true, pred_labels)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'F1-score: {f1score * 100:.2f}%')

    return pred_labels

def convert_prefetchdataset_to_numpy_arrays(prefetchdataset, data_type="spectrogram"):
    x_list = []
    y_list = []
    for batch in prefetchdataset:
        x_batch, y_batch = batch
        # We need this line for spectrogram
        if data_type == "spectrogram":
            x_batch = tf.expand_dims(x_batch, axis=-1)
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

def evaluate_prediction(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="binary")
    f1score = f1_score(y_true, y_pred, average="binary")

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'F1-score: {f1score * 100:.2f}%')

def get_f1_scores_of_bootstarping_partitions(model, x_data, y_true, model_format, input_data_uint8_type, n_bootstrap=100, n_chosen_samples=100):
    f1_scores = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_chosen_samples, replace=True)
        if model_format == "keras":
            y_pred_prob = model.predict(x_data[indices], verbose=0)
            y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
        elif model_format == "tf_lite":
            y_pred = tf_lite_model_predict(model, x_data[indices], input_data_uint8_type)
        f1 = f1_score(y_true[indices], y_pred)
        f1_scores.append(f1)
    return f1_scores

def get_f1_scores_of_non_overlapping_partitions(model, x_data, y_true, model_format, input_data_uint8_type, n_partitions=10):
    partition_size = len(x_data) // n_partitions
    partitions = []

    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size
        partitions.append((np.arange(start, end)))

    f1_scores = []

    for indices in partitions:
        if model_format == "keras":
            y_pred_prob = model.predict(x_data[indices], verbose=0)
            y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
        elif model_format == "tf_lite":
            y_pred = tf_lite_model_predict(model, x_data[indices], input_data_uint8_type)
        f1 = f1_score(y_true[indices], y_pred)
        f1_scores.append(f1)

    return f1_scores

def tf_lite_model_predict(tf_lite_model, x_data, input_data_uint8_type=False):
    interpreter = tf.lite.Interpreter(model_content=tf_lite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred_data = []
    for i in range(len(x_data)):
        input_data = np.expand_dims(x_data[i], axis=0)
        if input_data_uint8_type == True:
            interpreter.set_tensor(input_details[0]["index"], tf.cast(input_data, tf.uint8))
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        y_pred_prob = interpreter.get_tensor(output_details[0]['index'])
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_pred_data.append(y_pred)
    return y_pred_data


def predict_and_print_full_results(model, x_data, y_true, model_format, input_data_uint8_type=False):
    # Predict
    if model_format == "keras":
        y_pred_prob = model.predict(x_data)
        y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
    elif model_format == "tf_lite":
        y_pred = tf_lite_model_predict(model, x_data, input_data_uint8_type)

    # Evaluate
    print("Basic assessment of the whole dataset (without any partitions):")
    evaluate_prediction(y_true, y_pred)

    print("\nDevide dataset into 10 non-overlapping patritions and get their mean F1-score")
    non_overlap_patritions_f1_scores = get_f1_scores_of_non_overlapping_partitions(model, x_data, y_true, model_format, input_data_uint8_type)
    print("Non-overlap mean F1-score: ", np.mean(non_overlap_patritions_f1_scores))

    print("\nGet 100 bootstrap samples from dataset with 100 samples each and get their mean F1-score")
    bootstrap_patritions_f1_scores = get_f1_scores_of_bootstarping_partitions(model, x_data, y_true, model_format, input_data_uint8_type)
    print("Bootstrap mean F1-score: ", np.mean(bootstrap_patritions_f1_scores))
    return (
        y_pred, 
        non_overlap_patritions_f1_scores, 
        bootstrap_patritions_f1_scores,
    )


def evaluate_time_of_prediction(model, x_data, y_true_data, model_format, input_data_uint8_type=False, show_prediction_evaluation=True):
    y_pred_data = []
    time_data = []
    for i in range(len(x_data)):
        # Add batch size channel to data point
        if model_format == "keras": 
            input_data = np.expand_dims(x_data[i], axis=0)
            start_time = time.time()
            y_pred_prob = model.predict([input_data], verbose=0)
            y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
            elapsed_time = time.time() - start_time
        elif model_format == "tf_lite":
            start_time = time.time()
            y_pred = tf_lite_model_predict(model, [x_data[i]], input_data_uint8_type)
            elapsed_time = time.time() - start_time
        time_data.append(elapsed_time)

        if show_prediction_evaluation:
            y_pred_data.append(y_pred[0])
            
    if show_prediction_evaluation:
        evaluate_prediction(y_true_data, y_pred_data)

    print("\nTime to make a prediction for a single data point")
    print(f"Mean: {round(np.mean(time_data), 3)} seconds")
    print(f"Max: {round(np.max(time_data), 3)} seconds")
    print(f"Min: {round(np.min(time_data), 3)} seconds")

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


def run_full_int_q_tflite_model(tflite_file, indices, x_data):
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(indices),), dtype=int)
  for i, test_image_index in enumerate(indices):
    test_data_point = x_data[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_data_point = test_data_point / input_scale + input_zero_point

    test_data_point = np.expand_dims(test_data_point, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_data_point)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

def full_int_model_predict(tflite_file, x_data):
  indices = range(len(x_data))
  predictions = run_full_int_q_tflite_model(tflite_file, indices, x_data)
  return predictions

def get_f1_scores_of_non_overlapping_partitions_full_int_q(tflite_file, x_data, y_true, n_partitions=10):
    partition_size = len(x_data) // n_partitions
    partitions = []

    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size
        partitions.append((np.arange(start, end)))

    f1_scores = []

    for indices in partitions:
        y_pred = full_int_model_predict(tflite_file, x_data[indices])
        f1 = f1_score(y_true[indices], y_pred)
        f1_scores.append(f1)

    return f1_scores

def get_f1_scores_of_bootstarping_partitions_full_int_q(tflite_file, x_data, y_true, n_bootstrap=100, n_chosen_samples=100):
    f1_scores = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_chosen_samples, replace=True)
        y_pred = full_int_model_predict(tflite_file, x_data[indices])
        f1 = f1_score(y_true[indices], y_pred)
        f1_scores.append(f1)
    return f1_scores