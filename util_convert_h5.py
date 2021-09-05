#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import sys
import tensorflow as tf

from model.parse_params import parse_params
from model.triplet_model_fn import model_fn

gpuNum = 1

if __name__ == "__main__":
    # read params path
    path = sys.argv[1]
    params = parse_params(path)
    params_path = pathlib.Path(path).parents[0]

    with tf.device(f'/device:GPU:{gpuNum}'):
        model = model_fn(params, is_training=False)
        model.load_weights(os.path.join(params_path, 'model'))
        model.summary()
        model.save(os.path.join(params_path, 'model.h5'))
