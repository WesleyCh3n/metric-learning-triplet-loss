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
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental import RandomFourierFeatures

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="./exp/0318svmDt100/svm_{epoch}",
        save_weights_only=True,
        #  save_best_only=True,  # Only save a model if `val_loss` has improved.
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
    layers.Dense(units=5),
    ])
#  Ds_model.load_weights("./exp/0317svm/svm_30")
#  Ds_model.summary()
#  hidden_layer = Ds_model.get_layer("random_fourier_features")
#  fc = tf.keras.layers.Dense(units=19)(hidden_layer.output)
#  model = Model(inputs=Ds_model.input, outputs=fc)

model.compile(
        optimizer='adam',
        loss='hinge',
        metrics=['categorical_accuracy']
        )
model.summary()


DsX = np.loadtxt("./predict/0316Ds14_train/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
DsY = np.loadtxt("./predict/0316Ds14_train/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
DsX_val = np.loadtxt("./predict/0316Ds14_val/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
DsY_val = np.loadtxt("./predict/0316Ds14_val/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
Dsx = np.loadtxt("./predict/0316Ds14_test/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
Dsy  = np.loadtxt("./predict/0316Ds14_test/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
DtX = np.loadtxt("./predict/0316Dt5_train/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
DtY = np.loadtxt("./predict/0316Dt5_train/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
DtX_val = np.loadtxt("./predict/0316Dt5_val/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
DtY_val = np.loadtxt("./predict/0316Dt5_val/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
Dtx = np.loadtxt("./predict/0316Dt5_test/cv-vecs.tsv",
        dtype=np.float32,
        delimiter='\t')
Dty  = np.loadtxt("./predict/0316Dt5_test/cv-metas.tsv",
        dtype=int,
        delimiter='\t')
#
#  X = np.r_[DsX,DtX]
#  Y = np.r_[DsY,DtY]
#  X_val = np.r_[DsX_val,DtX_val]
#  Y_val = np.r_[DsY_val,DtY_val]
#  x = np.r_[Dsx,Dtx]
#  y = np.r_[Dsy,Dty]
#
#  Y = keras.utils.to_categorical(Y)
#  Y_val = keras.utils.to_categorical(Y_val)
#  y = keras.utils.to_categorical(y)

Dt_percls = 2400
Dt_count = 110
cls = int(len(DtX)/Dt_percls)

## Few-shot
# Suffle each class
[np.random.shuffle(DtX[i*Dt_percls:(i+1)*Dt_percls]) for i in range(cls)]
# Choose the few-shot number
DtX_few = np.zeros((Dt_count*cls,128),dtype=np.float32)
DtY_few = np.zeros((Dt_count*cls),dtype=np.float32)
for i in range(cls):
    DtX_few[Dt_count*i:Dt_count*(i+1)] = DtX[i*Dt_percls:i*Dt_percls+Dt_count]
    DtY_few[Dt_count*i:Dt_count*(i+1)] = DtY[i*Dt_percls:i*Dt_percls+Dt_count]

DtY_few = keras.utils.to_categorical(DtY_few)
Dty = keras.utils.to_categorical(Dty)

model.fit(DtX_few, DtY_few, epochs=50, batch_size=32, validation_split=0.1, callbacks=[callbacks])
results = model.evaluate(Dtx, Dty, batch_size=128)
print("test loss, test acc:", results)
