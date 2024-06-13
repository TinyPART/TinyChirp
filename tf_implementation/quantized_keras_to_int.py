import tensorflow as tf
import numpy as np
from tensorflow import keras

# keras ver. 3.3.3
# tf ver. 2.16.1

def dummy_representative_dataset():
    for _ in range(100):
      data = np.random.rand(1,144000,1)
      yield [data.astype(np.float32)]
#      yield data


model_path = "./time_series_models_to_test_with_RIOT_ML/squeezenet_30%_time_series_sr_48000.keras"

keras_model = keras.models.load_model(model_path)

# workaround the issue mentioned by https://github.com/keras-team/keras-core/issues/746
tf_callable = tf.function(
      keras_model.call,
      autograph=False,
      input_signature=[tf.TensorSpec((1,144000,1), tf.float32)],
)
tf_concrete_function = tf_callable.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_concrete_function], tf_callable
)

#converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT] # applies PTQ on weights, if possible
converter.representative_dataset = dummy_representative_dataset
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

import pathlib

tflite_models_dir = pathlib.Path("./")

tflite_model_file = tflite_models_dir/"squeezenet_30%_time_series_sr_48000_full_int8_quantization.tflite"
tflite_model_file.write_bytes(tflite_model_quant)
