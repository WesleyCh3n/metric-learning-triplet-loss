params = {
    'n_epochs': 100,
    'n_class': 19,
    'n_class_per_batch': 19,
    'n_per_class': 10,
    'size': [224, 224],
    'margin': 0.7,
    'lr': 'default',
    'early_stopping': 20,

    'pretrained_weight': './exp/sample_experiment/baseline_softmax/model', # path to baseline_softmax
    'train_ds': '/home/ubuntu/dataset/CowFace19/train/',
    'save_every_n_epoch': 1
}
