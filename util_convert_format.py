#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import sys
import pathlib
import tensorflow as tf

from model.parse_params import parse_params
from model.triplet_model_fn import model_fn

gpuNum = 1

if __name__ == "__main__":
    # read params path
    path = sys.argv[1]
    params = parse_params(path)
    params_path = pathlib.Path(path).parents[0]

    # user input
    s = """Which format do you want to convert?
1. h5
2. tflite
Please type the number (1/2): """
    num = int(input(s))

    # convert format
    with tf.device(f'/device:GPU:{gpuNum}'):
        print("Loading model...")
        str_list = []
        model = model_fn(False, **params)
        model.load_weights(os.path.join(params_path, 'model'))
        model.summary(line_length=50,
                      print_fn=lambda x: str_list.append(x))

        output_str = "\n".join(str_list[-5:])
        print(output_str)
        if num == 1:
            saving_path = os.path.join(params_path, 'model.h5')
            print(f"Saving {saving_path}.")
            model.save(saving_path)
            print("Done!")

        elif num == 2:
            saving_path = os.path.join(params_path, 'model.tflite')
            print(f"Saving {saving_path}.")

            model._set_inputs(inputs=tf.random.normal(shape=(1,
                                                             params['size'][0],
                                                             params['size'][1],
                                                             3)))
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            open(saving_path, "wb").write(tflite_model)
            print("Done")
