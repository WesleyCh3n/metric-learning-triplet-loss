# Few-shot learning (FSL) Experment

[![WesleyCh3n - FSL](https://img.shields.io/badge/WesleyCh3n-FSL-2ea44f?logo=github)](https://github.com/WesleyCh3n/FSL)
![Python - >=3.6.9](https://img.shields.io/badge/Python->=3.6.9-informational?logo=Python) 
![Tensorflow - 2.2.0](https://img.shields.io/badge/Tensorflow-2.2.0-informational?logo=Tensorflow) 
[![hackmd-github-sync-badge](https://hackmd.io/ct3mDHTJR2CLHUrys-jv2A/badge)](https://hackmd.io/ct3mDHTJR2CLHUrys-jv2A)


## Overview


![](https://raw.githubusercontent.com/WesleyCh3n/MobileNetv2-CowFace-Extractor/main/img/flowchart.svg)


## Baseline Model Training

In `baseline model training`, we use `Source domain dataset` (
<img src="https://render.githubusercontent.com/render/math?math=D_s">
).

### 1. Weight Initialization: `Cross-Entropy Loss`

- Edit `exp/sample_experiment/baseline_softmax/params.py`

    ```python
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

- During training, to visualize triplet loss, hardest negative distance (*HND*) and hardest positive distance (*HPD*):

    ```bash=
    tensorboard --logdir <path/to/params.py>
    ```
    
- Start training:

    ```bash=
    python3 train_softmax2triplet.py <path/to/params.py>
    ```
    
## FSL Update Training

In `FSL update training`, we use `Target domain dataset` (
<img src="https://render.githubusercontent.com/render/math?math=D_t">
).

- Edit `exp/sample_experiment/fewshot-triplet/params.py`

    ```python=
    params = {
        'n_epochs': 100,
        'n_class': 23,  # TODO
        'n_class_per_batch': 23,  # TODO
        'n_per_class': 10,  # TODO
        'size': [224, 224],
        'margin': 0.7,  # TODO
        'lr': 'default',
        'early_stopping': 20,

        'pretrained_weight': '/path/to/baseline_triplet/model',  # TODO
        'train_ds': '/path/to/train_ds',  # TODO
        'save_every_n_epoch': 1
    }
    ```

- During training, to visualize triplet loss, hardest negative distance (*HND*) and hardest positive distance (*HPD*):

    ```bash=
    tensorboard --logdir <path/to/params.py>
    ```
    
- Start training:

    ```bash=
    python3 train_triplet_fine_tune.py <path/to/params.py>
    ```
###### tags: `FSL`, `Triplet loss`