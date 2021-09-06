import random
import pathlib

import numpy as np
import tensorflow as tf


def dataset_pipeline(is_training: bool, batch=True, **params):
    """
    Read and preprocess images then return dataset format and its length.

    Args:
        folder (str): data directory which should contain sub-directory for
            each class. Such as:
            '''
            .
            ├── data_dir/
            │   ├── Class_1/
            │   ├── Class_2/
            ...
            │   └── Class_N/

            '''
        params (dict): parameters dictionary
        is_training (bool): if dataset is going to be trained or not
        batch (bool): return batched dataset or not

    Return:
        ds (tf.data.Dataset): tensorflow dataset format
        ds_count (int): dataset length
    """
    def parse_filename(path):
        ds_root = pathlib.Path(path)
        filenames = list(ds_root.glob('*/*'))
        filenames = [str(path) for path in filenames]
        random.shuffle(filenames)
        label_names = sorted(item.name for item in ds_root.glob('*/')
                if item.is_dir())
        label_to_index = dict((name, index) for index, name
                in enumerate(label_names))
        labels = [label_to_index[pathlib.Path(path).parent.name]
                for path in filenames]
        return filenames, labels, len(filenames)

    def _parse_function(data: dict, size: list, is_training: bool):
        image_string = tf.io.read_file(data['img'])
        image = tf.image.decode_jpeg(
            image_string, channels=3, dct_method='INTEGER_ACCURATE')
        image = tf.cast(image, tf.float32)
        # brightness augmentation
        if is_training:
            delta = tf.random.uniform([], 0.3, 1.0)
        else:
            delta = 1.0
        image = ((image * delta / 255.0)-0.5)*2.0
        image = tf.image.resize(image, size)
        data['img'] = image
        return data

    def generate_dataset(f, l, is_training, **params):
        parse_fn = lambda d: _parse_function(d, params["size"], is_training)
        if batch:
            dataset = (
                tf.data.Dataset.from_tensor_slices({'img':f, 'label':l})
                .shuffle(len(f))
                .map(parse_fn, num_parallel_calls=4)
                .batch(params["batch_size"])
                .repeat(params['n_epochs'])
                .prefetch(1)
            )
        if not batch:
            dataset = (
                tf.data.Dataset.from_tensor_slices({'img':f, 'label':l})
                .map(parse_fn, num_parallel_calls=4)
            )
        return dataset

    folder = params['train_ds'] if is_training else params['test_ds']
    ds_filenames, ds_labels, ds_counts = parse_filename(folder)
    ds = generate_dataset(ds_filenames, ds_labels, is_training, **params)
    return ds, ds_counts

#  def dataset_pipeline_balance_label(folder: str, params: dict, is_training: bool):
def dataset_pipeline_balance_label(is_training: bool, batch=True, **params):
    """
    Read and preprocess images then return dataset format and its length. Also,
    Balance each class number per batch

    Args:
        folder (str): data directory which should contain sub-directory for
            each class. Such as:
            '''
            .
            ├── data_dir/
            │   ├── Class_1/
            │   ├── Class_2/
            ...
            │   └── Class_N/

            '''
        params (dict): parameters dictionary
        is_training (bool): if dataset is going to be trained or not

    Return:
        dataset (tf.data.Dataset): tensorflow dataset format
        total_num (int): dataset length
    """
    #===============================================================================
    # Source: edloper @ GitHub.com
    # (https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000)
    # Fix original tf.py_function `inp` and `Tout` can't pass or return dictionary
    # type
    def _dtype_to_tensor_spec(v):
        return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

    def _tensor_spec_to_dtype(v):
        return v.dtype if isinstance(v, tf.TensorSpec) else v

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

    def _parse_function(filename, l_dict:dict, size: list, is_training: bool):
        name = pathlib.Path(filename.numpy().decode('utf-8')).parent.name
        label = l_dict[name]

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3,
                                     dct_method='INTEGER_ACCURATE')
        image = tf.cast(image, tf.float32)
        # brightness augmentation
        if is_training:
            delta = tf.random.uniform([], 0.3, 1.0)
        else:
            delta = 1.0
        image = ((image * delta / 255.0)-0.5)*2.0
        image = tf.image.resize(image, size)
        return image, label

    folder = params['train_ds']
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
        inp=[x,l_dict,params["size"], is_training],
        Tout={'img':tf.float32, 'label':tf.int32}
    )
    dataset = (tf.data.experimental.choose_from_datasets(datasets, choice_ds)
        .shuffle(total_num)
        .map(parse_fn)
        .batch(params["n_class_per_batch"]*params["n_per_class"])
        .repeat(params['n_epochs'])
        .prefetch(1)
    )

    return dataset, total_num
