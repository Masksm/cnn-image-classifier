import tensorflow as tf

from data_loader import IMAGE_SIZE, data_augmentation


NUM_CLASSES = 6


def build_improved_cnn():
    """
    Build the improved CNN model.
    """

    model = tf.keras.Sequential([

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        tf.keras.layers.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ),

        # ----------------------------------------------------
        # Normalize pixel values
        # ----------------------------------------------------

        tf.keras.layers.Rescaling(1.0 / 255),

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        data_augmentation,

        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 3
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 4
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),

        # ----------------------------------------------------
        # Classification layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(128),

        tf.keras.layers.Activation("relu"),

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        tf.keras.layers.Dropout(0.5),

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


def compile_improved_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_improved_cnn()

    model = compile_improved_model(model)

    print("\n" + "=" * 70)
    print("IMPROVED CNN MODEL")
    print("=" * 70)

    model.summary()