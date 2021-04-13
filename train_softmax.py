import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 1

import math
import datetime
import sys
import time
from collections import deque
import numpy as np
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

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from model.input_fn import dataset_pipeline
from model.softmax_model_fn import model_fn
import importlib.util


if __name__ == "__main__":
    # read params path
    params_path = sys.argv[1]

    # load parameters
    spec = importlib.util.spec_from_file_location(
        'params', os.path.join(params_path, 'params.py'))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    params = loader.params

    # create dataset
    train_ds, train_count = dataset_pipeline(params['train_ds'], params)
    test_ds, test_count = dataset_pipeline(params['test_ds'], params)

    # build model
    model = model_fn(params, is_training=True)
    model.summary()

    log_dir = os.path.join(params_path, "logs/",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir,
        update_freq='batch',
        profile_batch=0,
        histogram_freq=1
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params_path, "model"),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
        verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['early_stopping']
    )

    # start training
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(
        train_ds,
        steps_per_epoch=math.ceil(train_count/params['batch_size']),
        epochs=params['n_epochs'],
        validation_data=test_ds,
        callbacks=[tensorboard_callback, cp_callback, es_callback]
    )
