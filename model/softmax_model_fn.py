import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


train_loss_tracker = tf.keras.metrics.Mean()
val_loss_tracker = tf.keras.metrics.Mean()
train_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

class CustomModel(tf.keras.Model):
    """
    Inherited from `tf.keras.Model`.
    Custom training step, test step, metrics.
    self.compiled_loss is SparseCategoricalCrossentropy.

    metrics include {train loss, val loss, train acc, val acc}
    """
    def train_step(self, data):
        x, y = data['img'], data['label']

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        train_loss_tracker.update_state(loss)
        train_acc_tracker.update_state(y, y_pred)
        return {"loss": train_loss_tracker.result(),
                "acc": train_acc_tracker.result()}
    def test_step(self, data):
        x, y = data['img'], data['label']
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compiled_loss(y, y_pred)
        # Update the metrics.
        val_loss_tracker.update_state(loss)
        val_acc_tracker.update_state(y, y_pred)
        return {"loss": val_loss_tracker.result(),
                "acc": val_acc_tracker.result()}
    @property
    def metrics(self):
        return [train_loss_tracker, val_loss_tracker,
                train_acc_tracker, val_acc_tracker]

def model_fn(params, is_training=True):
    """
    Create base model with MobileNetV2 + Dense layer (n class).
    Wrap up with CustomModel process.

    Args:
        params (dict): parameters dictionary
        is_training (bool): if it is going to be trained or not
    """
    baseModel = MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling="avg")
    fc = tf.keras.layers.Dense(
        params['n_class'], activation="softmax",
        name="softmax_layer")(baseModel.output)

    model = CustomModel(inputs=baseModel.input, outputs=fc)

    # If it is not training mode
    if not is_training:
        model.trainable = False
    return model

