#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
gpuNum = 0

import sys
import pathlib
from tqdm import tqdm
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
from tensorboard.plugins import projector
from model.input_fn_none_batch import none_batch_dataset_pipeline
from model.triplet_model_fn import model_fn
import importlib.util


def gen_ds(test_ds_path, params, total_class=19):
    test_ds, _ = none_batch_dataset_pipeline(test_ds_path, params)
    vecs = np.empty((0,128),np.float)
    metas = np.empty((0),np.float)
    for cls in tqdm(range(total_class)):
        ds = test_ds.filter(lambda data: data['label'] == cls).batch(300)
        for data in ds:
            feat = model.predict(data['img'])
            vecs = np.append(vecs, feat, axis=0)
            metas = np.append(metas, data['label'].numpy())
    return vecs, metas


if __name__ == "__main__":
    # read params path
    params_path = sys.argv[1]

    # load parameters
    spec = importlib.util.spec_from_file_location(
        "params", os.path.join(params_path, 'params.py'))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    params = loader.params

    # logdir
    logdir = pathlib.Path(params_path).joinpath('feats')
    logdir.mkdir(parents=True, exist_ok=True)

    # build model
    model = model_fn(params, is_training=False)
    model.load_weights(os.path.join(params_path, 'model'))

    # create embeddings, metadata
    vecs, metas = gen_ds("/home/ubuntu/dataset/DsDt_test300/", params)
    np.savetxt(str(logdir.joinpath("vec300.tsv")),
            vecs, fmt='%.8f', delimiter='\t')
    np.savetxt(str(logdir.joinpath("meta300.tsv")),
            metas, fmt='%i', delimiter='\t')

    vecs, metas = gen_ds("/home/ubuntu/dataset/test_100/", params)
    np.savetxt(str(logdir.joinpath("vec100.tsv")),
            vecs, fmt='%.8f', delimiter='\t')
    np.savetxt(str(logdir.joinpath("meta100.tsv")),
            metas, fmt='%i', delimiter='\t')

    # TODO: Not working
    #  embedding_var = tf.Variable(vecs, name='300_cow_face')
    #  checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    #  checkpoint.save(os.path.join(logdir, "300embedding.ckpt"))
    #  # Set up config
    #  config = projector.ProjectorConfig()
    #  embedding = config.embeddings.add()
    #  embedding.tensor_name = embedding_var.name
    #  embedding.metadata_path = 'meta300.tsv'
    #  projector.visualize_embeddings(logdir, config)
    #
