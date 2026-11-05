import tensorflow as tf

from data_loader_v2 import IMAGE_SIZE


NUM_CLASSES = 6

BASE_MODEL_PATH = "model/best_inception_transfer.keras"


def build_finetuning_model():

    # ========================================================
    # Load the already-trained transfer-learning model
    # ========================================================

    model = tf.keras.models.load_model(
        BASE_MODEL_PATH
    )

    # ========================================================
    # Find the InceptionV3 base model
    # ========================================================

    base_model = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            base_model = layer
            break

    if base_model is None:

        raise RuntimeError(
            "Could not find the InceptionV3 base model."
        )

    # ========================================================
    # Freeze everything first
    # ========================================================

    base_model.trainable = True

    # Freeze the lower layers
    # and fine-tune only the upper layers.

    fine_tune_from = 249

    for layer in base_model.layers[:fine_tune_from]:

        layer.trainable = False

    for layer in base_model.layers[fine_tune_from:]:

        layer.trainable = True

    # ========================================================
    # Keep BatchNormalization layers frozen
    # ========================================================

    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    return model


def compile_finetuning_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_finetuning_model()

    model = compile_finetuning_model(
        model
    )

    print("\n" + "=" * 70)
    print("INCEPTIONV3 FINE-TUNING MODEL")
    print("=" * 70)

    model.summary()

    trainable_parameters = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    non_trainable_parameters = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.non_trainable_weights
    )

    print(
        f"\nTrainable parameters:"
        f" {trainable_parameters:,}"
    )

    print(
        f"Non-trainable parameters:"
        f" {non_trainable_parameters:,}"
    )