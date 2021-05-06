import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import math
import datetime
import sys
import tensorflow as tf
from model.parse_params import parse_params
from model.input_fn import dataset_pipeline
from model.softmax_model_fn import model_fn

gpuNum = 1

if __name__ == "__main__":
    # read params path
    params_path = sys.argv[1]
    params = parse_params(params_path)

    with tf.device(f'/device:GPU:{gpuNum}'):
        # create dataset
        train_ds, train_count = dataset_pipeline(params['train_ds'], params, True)
        val_ds, val_count = dataset_pipeline(params['test_ds'], params, False)

        # build model
        model = model_fn(params, is_training=True)
        model.summary()

        log_dir = os.path.join(params_path, "logs/",
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir,
            update_freq='batch',
            profile_batch=0,
            histogram_freq=1
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(params_path, "model"),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params['early_stopping']
        )

        # start training
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy())
        model.fit(
            train_ds,
            steps_per_epoch=math.ceil(train_count/params['batch_size']),
            epochs=params['n_epochs'],
            validation_data=val_ds,
            validation_steps=math.ceil(val_count/params['batch_size']),
            callbacks=[tensorboard_callback, cp_callback, es_callback]
        )
