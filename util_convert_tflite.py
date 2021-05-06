#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from model.parse_params import parse_params

gpuNum = '1'


if __name__ == "__main__":
    # read params path
    params_path = sys.argv[1]
    params = parse_params(params_path)

    with tf.device(f'/device:GPU:{gpuNum}'):
        baseModel = MobileNetV2(include_top=False,
                                weights=None,
                                input_shape=(224, 224, 3),
                                pooling="avg")
        fc = tf.keras.layers.Dense(128,
                                   activation=None,
                                   name="embeddings")(baseModel.output)
        l2 = tf.math.l2_normalize(fc)

        model = Model(inputs=baseModel.input, outputs=l2)
        model.load_weights(os.path.join(params_path, "model")).expect_partial()


        model._set_inputs(inputs=tf.random.normal(shape=(1, params['size'][0], params['size'][1], 3)))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        filepath = os.path.join(params_path, "model.tflite")
        open(filepath, "wb").write(tflite_model)
        print("Done")
