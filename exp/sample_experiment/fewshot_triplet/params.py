params = {
    'n_epochs': 1000,
    'n_class': 23,
    'n_class_per_batch': 23,
    'n_per_class': 10,
    'size': [224, 224],
    'margin': 0.7,
    'lr': 0.00001,
    'early_stopping': 100,

    'pretrained_weight': './exp/sample_experiment/baseline_triplet/model',
    'train_ds': '/home/ubuntu/dataset/CowFaceAdd100/',
    'test_ds': '/home/ubuntu/dataset/CowFaceAdd100/',
    'save_every_n_epoch': 1
}
