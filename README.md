# Few-shot learning (FSL) Experment

[![hackmd-github-sync-badge](https://hackmd.io/ct3mDHTJR2CLHUrys-jv2A/badge)](https://hackmd.io/ct3mDHTJR2CLHUrys-jv2A)


## Overview


![](https://raw.githubusercontent.com/WesleyCh3n/MobileNetv2-CowFace-Extractor/main/img/flowchart.svg)


## Baseline Model Training

In `baseline model training`, we use `Source domain dataset` ($D_s$).

### 1. Weight Initialization: `Cross-Entropy Loss`

- Edit `exp/sample_experiment/baseline_softmax/params.py`

    ```python=
        params = {
        'n_epochs': 50,
        'n_class': 19,  # TODO
        'size': [224, 224],
        'batch_size': 64,
        'lr': 'default',
        'early_stopping': 5,

        'train_ds': '/path/to/train_ds',  # TODO
        'test_ds': '/path/to/test_ds',  # TODO
        'save_every_n_epoch': 1
    }
    ```

- Start training:

    ```bash=
    python3 train_softmax.py <path/to/params.py>
    ```

- During training, to visualize loss and accuracy:

    ```bash=
    tensorboard --logdir <path/to/params.py>
    ```


- After training complete:
    - Model Checkpoint directory is same as `path/to/params.py`

### 2. Baseline Feature Extractor: `Triplet Loss`

- Edit `exp/sample_experiment/baseline_triplet/params.py`

    ```python=
    params = {
        'n_epochs': 100,
        'n_class': 19,  # TODO
        'n_class_per_batch': 19,  # TODO
        'n_per_class': 10,  # TODO
        'size': [224, 224],
        'margin': 0.7,  # TODO
        'lr': 'default',
        'early_stopping': 20,

        'pretrained_weight': '/path/to/baseline_softmax/model',  # TODO
        'train_ds': '/path/to/train_ds',  # TODO
        'save_every_n_epoch': 1
    }
    ```

###### tags: `FSL`, `Triplet loss`