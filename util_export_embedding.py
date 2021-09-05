#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import sys
import pathlib
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorboard.plugins import projector
from model.parse_params import parse_params
from model.triplet_model_fn import model_fn
from model.input_fn import dataset_pipeline

gpuNum = 3

def gen_ds(test_ds_path, params, total_class):
    test_ds, _ = dataset_pipeline(test_ds_path, params,
                                  is_training=False, batch=False)
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
    dataset = '/home/ubuntu/dataset/new_class/'
    output_dir_name = 'feat'

    # read params path
    path = sys.argv[1]
    params = parse_params(path)
    params_path = pathlib.Path(path).parents[0]

    # logdir
    logdir = pathlib.Path(params_path).joinpath(output_dir_name)
    logdir.mkdir(parents=True, exist_ok=True)

    # build model
    model = model_fn(params, is_training=False)
    model.load_weights(os.path.join(params_path, 'model'))

    # create embeddings, metadata
    with tf.device(f'/device:GPU:{gpuNum}'):
        vecs, metas = gen_ds(dataset, params, params['n_class'])
    np.savetxt(str(logdir.joinpath("vec.tsv")),
            vecs, fmt='%.8f', delimiter='\t')
    np.savetxt(str(logdir.joinpath("meta.tsv")),
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
