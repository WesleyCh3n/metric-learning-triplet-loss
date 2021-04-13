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

    # create dataset
    train_ds, train_count = dataset_pipeline(params['train_ds'], params)
    test_ds, test_count = dataset_pipeline(params['test_ds'], params)

    # build model
    baseModel = MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling="avg")
    fc = tf.keras.layers.Dense(
        params['num_classes'], activation="softmax",
        name="dense_final")(baseModel.output)
    model = Model(inputs=baseModel.input, outputs=fc)
    #  model.summary()

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    logdir = os.path.join(params_path, "logs/",
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()

    def train_step(images, labels, step):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(y_true=labels, y_pred=predictions)
        return loss

    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_accuracy.update_state(labels, predictions)
        return t_loss

    ts_total = time.time()
    total_step = 0
    min_loss = 10 # Rather large loss
    early_stopping = 100
    loss_history = deque(maxlen=early_stopping + 1)
    # start training
    for epoch in tf.range(params['n_epochs']):
        ts = time.time()
        for step, data in enumerate(train_ds):
            total_step += 1
            loss = train_step(data['img'], data['label'], total_step)
            template = "Epoch: {}/{}, step: {}/{}, loss: {:.5f}, acc: {:.5f}"
            print(template.format(epoch,
                  params['n_epochs'],
                  step,
                  math.ceil(train_count/
                  params['batch_size']),
                  loss.numpy(),
                  float(train_accuracy.result())), end="\r")
            tf.summary.scalar('train_loss', loss, step=total_step)
            tf.summary.scalar('train_accuracy',
                              train_accuracy.result(), step=total_step)

        for data in test_ds:
            test_loss = test_step(data['img'], data['label'])
        tf.summary.scalar('test_loss', test_loss, step=total_step)
        tf.summary.scalar('test_accuracy',
                          test_accuracy.result(), step=total_step)
        loss_history.append(test_loss.numpy())
        print(f"\ntest accuracy: {test_accuracy.result():.2f}")

        train_accuracy.reset_states()
        test_accuracy.reset_states()

        if epoch % params['save_every_n_epoch'] == 0:
            model.save_weights(os.path.join(params_path, "model"),
                               save_format='tf')
        #      if test_loss.numpy() < min_loss:
        #          print(f"model save. Improve from {min_loss:.4f} to {test_loss.numpy():.4f}")
        #          model.save_weights(os.path.join(params_path, "model"),
        #                             save_format='tf')
        #          min_loss = test_loss.numpy()
        #
        #  if len(loss_history) > early_stopping:
        #      if loss_history.popleft() < min(loss_history):
        #          print(f'Early stopping. No validation loss '
        #                f'improvement in {early_stopping} epochs.')
        #          break
        print(f"Epoch: {epoch}, spend {time.time()-ts:.2f}sec")

    print(f"Spend {time.time()-ts_total:.2f}sec")
