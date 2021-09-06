import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from model.triplet_loss import batch_hard_triplet_loss

loss_tracker = tf.keras.metrics.Mean(name="loss")
HPD_tracker = tf.keras.metrics.Mean(name="hardest_positive_dist")
HND_tracker = tf.keras.metrics.Mean(name="hardest_negative_dist")


class CustomModel(tf.keras.Model):
    """
    Inherited from `tf.keras.Model`.
    Custom training step, metrics.
    self.compiled_loss is triplet loss.

    metrics include {loss,
                     hardest positive distance (HPD),
                     hardest negative distance (HND)}
    """
    def __init__(self, margin=None, **kwargs):
        self.margin = margin
        super(CustomModel, self).__init__(**kwargs)

    def train_step(self, data):
        x, y = data['img'], data['label']

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss, hpd, hnd = batch_hard_triplet_loss(y,
                                                     y_pred,
                                                     self.margin)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        HPD_tracker.update_state(hpd)
        HND_tracker.update_state(hnd)
        return {"loss": loss_tracker.result(),
                "HPD": HPD_tracker.result(),
                "HND": HND_tracker.result()}
    @property
    def metrics(self):
        return [loss_tracker, HPD_tracker, HND_tracker]

def model_fn(is_training=True, **params):
    """
    Create feature extractor model with MobileNetV2 + Dense layer (128).
    Wrap up with CustomModel process.

    Args:
        is_training (bool): if it is going to be trained or not
        params: keyword arguments (parameters dictionary)
    """
    baseModel = MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling="avg"
    )

    # embeddings layer
    fc = tf.keras.layers.Dense(
        128, activation=None, name="fully_conneted_128"
    )(baseModel.output)
    l2 = tf.math.l2_normalize(fc)
    model = CustomModel(inputs=baseModel.input,
                        outputs=l2, margin=params["margin"])

    # If it is not training mode
    if not is_training:
        model.trainable = False
    return model

def transfer_model_fn(is_training=True, **params):
    """
    Load MobileNetV2 + Dense layer (n class) weight, replace
    Dense layer (n class) with Dense layer (128).
    Wrap up with CustomModel process. Then train with triplet loss.

    Args:
        is_training (bool): if it is going to be trained or not
        params: keyword arguments (parameters dictionary)
    """
    baseModel = MobileNetV2(
        include_top=False, weights=None,
        input_shape=(224, 224, 3), pooling="avg"
    )
    fc = tf.keras.layers.Dense(
        params['n_class'], activation="softmax",
        name="softmax_layer")(baseModel.output)
    model = Model(inputs=baseModel.input, outputs=fc)
    model.load_weights(params['pretrained_weight']).expect_partial()

    preLayer = model.get_layer("global_average_pooling2d")
    fc = tf.keras.layers.Dense(
        128, activation=None, name="fully_conneted_128"
    )(preLayer.output)
    l2 = tf.math.l2_normalize(fc)
    model = CustomModel(inputs=baseModel.input,
                        outputs=l2, margin=params["margin"])

    # If it is not training mode
    if not is_training:
        model.trainable = False
    return model

def fine_tune_model_fn(is_training=True, **params):
    """
    Load MobileNetV2 + Dense layer (128) weight, freeze weight to
    block_16_depthwise_relu layer, only train last 2 layer
    Wrap up with CustomModel process. Then train with triplet loss.

    Args:
        is_training (bool): if it is going to be trained or not
        params: keyword arguments (parameters dictionary)
    """
    base_model = MobileNetV2(
        include_top=False, weights=None,
        input_shape=(224, 224, 3), pooling="avg"
    )

    freeze_layer = base_model.get_layer("block_16_depthwise_relu")
    freeze_model = Model(inputs=base_model.input, outputs=freeze_layer.output)
    freeze_model.trainable = False
    freeze_model = Model(inputs=freeze_model.input, outputs=base_model.output)

    # embeddings layer
    fc = tf.keras.layers.Dense(
        128, activation=None, name="fully_conneted_128"
    )(freeze_model.output)
    l2 = tf.math.l2_normalize(fc)
    model = CustomModel(inputs=freeze_model.input,
                outputs=l2,
                margin=params["margin"])

    # If it is not training mode
    if not is_training:
        model.trainable = False
    return model
