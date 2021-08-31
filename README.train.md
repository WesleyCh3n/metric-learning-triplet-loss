# FSL Training Code

[![WesleyCh3n - FSL](https://img.shields.io/badge/WesleyCh3n-FSL-2ea44f?logo=github)](https://github.com/WesleyCh3n/FSL)
[![hackmd-github-sync-badge](https://hackmd.io/Y-MdlpaZRBKtj-nBfWyyJw/badge)](https://hackmd.io/Y-MdlpaZRBKtj-nBfWyyJw)

## Overview

- Dataset: `tf.data.Dataset`
- Model: `tf.keras.models.Model`
    - Model Structure
    - Training loop
        - Loss
        - Optirmizer: `tf.keras.optimizers`
- Callbacks
    - Tensorboard: `tf.keras.callbacks.TensorBoard`
    - Model checkpoint: `tf.keras.callbacks.ModelCheckpoint`
    - Early stopping: `tf.keras.callbacks.EarlyStopping`

---

## Dataset

### `./model/input_fn.py`

- **def dataset_pipeline(folder: str, params: dict, is_training: bool, batch=True)**

    ```python=
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
        dataset (tf.data.Dataset): tensorflow dataset format
        total_num (int): dataset length
    """
    ```

- **def dataset_pipeline_balance_label(folder: str, params: dict, is_training: bool)**

    ```python=
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
    ```

## Model

### `./model/softmax_model_fn.py`

- **class CustomModel(tf.keras.Model)**

    ```python=
    """
    Inherited from `tf.keras.Model`.
    Custom training step, test step, metrics.
    self.compiled_loss is SparseCategoricalCrossentropy.

    metrics include {train loss, val loss, train acc, val acc}
    """
    ```

- **def model_fn(params, is_training)**

    ```python=
    """
    Create base model with MobileNetV2 + Dense layer (n class).
    Wrap up with CustomModel process.

    Args:
        params (dict): parameters dictionary
        is_training (bool): if it is going to be trained or not
    """
    ```

### `./model/triplet_model_fn.py`

- **class CustomModel(tf.keras.Model)**

    ```python=
    """
    Inherited from `tf.keras.Model`.
    Custom training step, metrics.
    self.compiled_loss is triplet loss.

    metrics include {loss,
                     hardest positive distance (HPD),
                     hardest negative distance (HND)}
    """
    ```

- **def model_fn(params, is_training)**

    ```python=
    """
    Create feature extractor model with MobileNetV2 + Dense layer (128).
    Wrap up with CustomModel process.

    Args:
        params (dict): parameters dictionary
        is_training (bool): if it is going to be trained or not
    """
    ```

- **def transfer_model_fn(params, is_training)**

    ```python=
    """
    Load MobileNetV2 + Dense layer (n class) weight, replace
    Dense layer (n class) with Dense layer (128).
    Wrap up with CustomModel process. Then train with triplet loss.

    Args:
        params (dict): parameters dictionary
        is_training (bool): if it is going to be trained or not
    """
    ```

- **def fine_tune_model_fn(params, is_training)**

    ```python=
    """
    Load MobileNetV2 + Dense layer (128) weight, freeze weight to
    block_16_depthwise_relu layer, only train last 2 layer
    Wrap up with CustomModel process. Then train with triplet loss.

    Args:
        params (dict): parameters dictionary
        is_training (bool): if it is going to be trained or not

    """
    ```

### `./model/triplet_loss.py`

- **def batch_hard_triplet_loss(labels, embeddings, margin, squared=False)**

    ```python=
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    ```

## Callbacks

- `tf.keras.callbacks.TensorBoard`: Record metrics
- `tf.keras.callbacks.ModelCheckpoint`: Save model checkpoint
- `tf.keras.callbacks.EarlyStopping`: Early stop training


###### tags: `FSL`