
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from config import NUM_CLASSES


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_multilabel_cxr")
    return model, base


def unfreeze_top_layers(base_model, num_layers=30):
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    return base_model


def weighted_binary_crossentropy(pos_weights):
    weights_tensor = tf.constant(list(pos_weights.values()), dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce = -(weights_tensor * y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(bce)

    return loss_fn
