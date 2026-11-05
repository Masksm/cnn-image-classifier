import tensorflow as tf

from data_loader_v2 import IMAGE_SIZE


NUM_CLASSES = 6


def build_transfer_model():

    # --------------------------------------------------------
    # Load pretrained InceptionV3
    # --------------------------------------------------------

    base_model = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # --------------------------------------------------------
    # Freeze pretrained layers
    # --------------------------------------------------------

    base_model.trainable = False

    # --------------------------------------------------------
    # Build classification model
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    # InceptionV3 requires its own preprocessing
    x = tf.keras.applications.inception_v3.preprocess_input(
        inputs
    )

    # Extract pretrained features
    x = base_model(
        x,
        training=False
    )

    # Convert feature maps into a feature vector
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification layer
    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    # Reduce overfitting
    x = tf.keras.layers.Dropout(
        0.5
    )(x)

    # Six output classes
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model


def compile_transfer_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_transfer_model()

    model = compile_transfer_model(
        model
    )

    print("\n" + "=" * 70)
    print("INCEPTIONV3 TRANSFER LEARNING MODEL")
    print("=" * 70)

    model.summary()

    print("\nTrainable parameters:")
    print(
        sum(
            tf.keras.backend.count_params(
                weight
            )
            for weight in model.trainable_weights
        )
    )

    print("\nNon-trainable parameters:")
    print(
        sum(
            tf.keras.backend.count_params(
                weight
            )
            for weight in model.non_trainable_weights
        )
    )