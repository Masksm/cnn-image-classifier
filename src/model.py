import tensorflow as tf

from data_loader import IMAGE_SIZE, data_augmentation


NUM_CLASSES = 6


def build_cnn_model():
    """
    Build and return the CNN image-classification model.
    """

    model = tf.keras.Sequential([
        
        # Input
        tf.keras.layers.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ),

        # Normalize pixel values from [0, 255] to [0, 1]
        tf.keras.layers.Rescaling(1.0 / 255),

        # Data augmentation
        data_augmentation,

        # Convolution Block 1
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolution Block 2
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolution Block 3
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convert feature maps into a single vector
        tf.keras.layers.Flatten(),

        # Classification layer
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation("relu"),

        # Reduce overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


def compile_model(model):
    """
    Configure the model for training.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_cnn_model()

    model = compile_model(model)

    print("\n" + "=" * 70)
    print("CNN MODEL ARCHITECTURE")
    print("=" * 70)

    model.summary()