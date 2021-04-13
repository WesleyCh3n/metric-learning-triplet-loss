#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 0

import math
import datetime
import sys
import time
import tensorflow as tf
from collections import deque
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

from model.balance_input_fn import dataset_pipeline
from model.triplet_loss import batch_hard_triplet_loss
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

    # dataset
    train_ds, train_count = dataset_pipeline(params['train_ds'], params)

    # create model
    baseModel = MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling="avg"
    )

    fc = tf.keras.layers.Dense(
        128, activation=None, name="embeddings"
    )(baseModel.output)

    l2 = tf.math.l2_normalize(fc)
    model = Model(inputs=baseModel.input, outputs=l2)
    #  model.summary()

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss, hpd, hnd = batch_hard_triplet_loss(labels,
                                                     predictions,
                                                     params['margin'])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, hpd, hnd

    logdir = os.path.join(params_path, "logs/",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir+'/train')
    file_writer.set_as_default()

    ts_total = time.time()
    total_step = 0
    min_loss = 10 # Rather large loss
    early_stopping = params['early_stopping']
    loss_history = deque(maxlen=early_stopping + 1)
    # start training
    for epoch in range(params['n_epochs']):
        ts = time.time()
        for step, data in enumerate(train_ds):
            total_step += 1
            loss, hpd, hnd = train_step(data['img'], data['label'])
            template = "Epoch: {}/{}, step: {}/{}, loss: {:.5f}"
            print(template.format(epoch,
                  params['n_epochs'],
                  step,
                  math.ceil(train_count/
                  (params['n_class_per_batch']*params['n_per_class'])),
                  loss.numpy()), end="\r")
            tf.summary.scalar('loss', loss, step=total_step)
            tf.summary.scalar("hardest_positive_dist", hpd, step=total_step)
            tf.summary.scalar("hardest_negative_dist", hnd, step=total_step)

        if epoch % params["save_every_n_epoch"] == 0:
            if loss.numpy() < min_loss:
                print(f"\nmodel save. Improve from {min_loss:.4f} to {loss.numpy():.4f}")
                model.save_weights(os.path.join(params_path, "model"),
                                   save_format='tf')
                min_loss = loss.numpy()

        if len(loss_history) > early_stopping:
            if loss_history.popleft() < min(loss_history):
                print(f'\nEarly stopping. No validation loss '
                      f'improvement in {early_stopping} epochs.')
                break
        print(f"Epoch: {epoch}, spend {time.time()-ts:.2f}sec")

    print(f"Spend {time.time()-ts_total:.2f}sec")
