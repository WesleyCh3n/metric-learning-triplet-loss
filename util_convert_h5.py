#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 2

import sys
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[gpuNum], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpuNum], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

from model.triplet_model_fn import model_fn
import importlib.util


if __name__ == "__main__":
    # read params path
    params_path = sys.argv[1]

    # load parameters
    spec = importlib.util.spec_from_file_location(
        "params", os.path.join(params_path, 'params.py'))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    params = loader.params

    model = model_fn(params, is_training=False)
    model.load_weights(os.path.join(params_path, 'model'))
    model.summary()
    model.save(os.path.join(params_path, 'model.h5'))
