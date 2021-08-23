#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import sys
import math
import pathlib
import datetime
import tensorflow as tf

from model.parse_params import parse_params
from model.balance_input_fn import dataset_pipeline
from model.triplet_model_fn import transfer_model_fn

gpuNum = 1


if __name__ == "__main__":
    # read params path
    path = sys.argv[1]
    params = parse_params(path)
    params_path = pathlib.Path(path).parents[0]

    with tf.device(f'/device:GPU:{gpuNum}'):
        # dataset
        train_ds, train_count = dataset_pipeline(params['train_ds'], params, True)

        model = transfer_model_fn(params, is_training=True)
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
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=params['early_stopping']
        )

        # start training
        model.compile(optimizer="adam")
        model.fit(
            train_ds,
            steps_per_epoch=math.ceil(
                train_count/(params['n_class_per_batch']*params['n_per_class'])
            ),
            epochs=params['n_epochs'],
            callbacks=[tensorboard_callback, cp_callback, es_callback]
        )
