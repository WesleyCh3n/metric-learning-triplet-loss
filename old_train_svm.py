#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 1

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:  # Restrict TensorFlow to only use the first GPU
    tf.config.experimental.set_visible_devices(gpus[gpuNum], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpuNum], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="./exp/0317svm/svm_{epoch}",
        save_weights_only=True,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]

model = keras.Sequential([
    keras.Input(shape=(128,)),
    RandomFourierFeatures(
        output_dim=4096,
        scale=10.,
        kernel_initializer='gaussian'),
    layers.Dense(units=14),
    ])
model.compile(
        optimizer='adam',
        loss='hinge',
        metrics=['categorical_accuracy']
        )
model.summary()

X = np.loadtxt("./predict/0316Ds14_train/cv-vecs.tsv",  # 33600 sample
        dtype=np.float32,
        delimiter='\t')
Y = np.loadtxt("./predict/0316Ds14_train/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
X_val = np.loadtxt("./predict/0316Ds14_val/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
Y_val = np.loadtxt("./predict/0316Ds14_val/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
x = np.loadtxt("./predict/0316Ds14_test/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
y  = np.loadtxt("./predict/0316Ds14_test/cv-metas.tsv",
        dtype=int,
        delimiter='\t')

Y = keras.utils.to_categorical(Y)
Y_val = keras.utils.to_categorical(Y_val)
y = keras.utils.to_categorical(y)

model.fit(X, Y, epochs=50, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callbacks])
results = model.evaluate(x, y, batch_size=128)
print("test loss, test acc:", results)
