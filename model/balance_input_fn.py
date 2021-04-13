import os

import numpy as np
import tensorflow as tf
import pathlib
import random


#===============================================================================
# Source: edloper @ GitHub.com
# (https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000)
# Fix original tf.py_function `inp` and `Tout` can't pass or return dictionary
# type
def new_py_function(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,
                expand_composites=True)
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)
    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name)
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout,
          expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out

def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v
#===============================================================================

def _choice_function(n_class:int, n_per_class:int, n_class_per_batch:int):
    while True:
        # Sample the labels that will compose the batch
        labels = np.random.choice(range(n_class),
                                  n_class_per_batch,
                                  replace=False)
        for label in labels:
            for _ in range(n_per_class):
                yield label

def _parse_function(filename, l_dict:dict, size: list):
    name = pathlib.Path(filename.numpy().decode('utf-8')).parent.name
    label = l_dict[name]

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3,
                                 dct_method='INTEGER_ACCURATE')
    image = tf.cast(image, tf.float32)
    image = ((image / 255.0)-0.5)*2.0
    image = tf.image.resize(image, size)
    return image, label

def dataset_pipeline(folder, params):
    r_path = pathlib.Path(folder)
    img_dirs = list(r_path.iterdir())
    total_num = len(list(r_path.glob('*/*')))
    # create label dict to num
    l_names = sorted(i.name for i in r_path.glob('*/') if i.is_dir())
    l_dict = dict((n, i) for i, n in enumerate(l_names))
    print("label dict: ", l_dict)

    #load filenames each class to a dataset list
    datasets = [tf.data.Dataset.list_files(f"{d}/*.jpg") for d in img_dirs]
    datasets = [dataset.shuffle(10000) for dataset in datasets]

    gen_fn = lambda :_choice_function(
        params["n_class"],
        params["n_per_class"],
        params["n_class_per_batch"]
    )
    choice_ds = tf.data.Dataset.from_generator(gen_fn, tf.int64)

    parse_fn = lambda x: new_py_function(
        func=_parse_function,
        inp=[x,l_dict,params["size"]],
        Tout={'img':tf.float32, 'label':tf.int32}
    )
    dataset = (tf.data.experimental.choose_from_datasets(datasets, choice_ds)
        .map(parse_fn)
        .batch(params["n_class_per_batch"]*params["n_per_class"])
        .repeat(params['n_epochs'])
        .prefetch(1)
    )

    return dataset, total_num
