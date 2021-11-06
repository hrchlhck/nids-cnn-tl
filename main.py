from pathlib import Path
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
from tensorflow import keras

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

def main() -> None:       
    width, height = 128, 128
    batch_size = 32
    input_tensor = Input(shape=(width, height, 3))

    data_dir = Path('data/image/2012/01/')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(width, height),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(width, height),
        batch_size=batch_size)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    class_names = train_ds.class_names
    print(class_names)

    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    model.trainable = False

    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = model(input_tensor, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(input_tensor, outputs)

    metrics = ['accuracy']
    model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[keras.metrics.BinaryAccuracy()])

    history = model.fit(train_ds, validation_data=val_ds, epochs=10, use_multiprocessing=True, workers=4)

    test_loss, test_acc = model.evaluate(val_ds, verbose=2)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    ax.plot(history.history['binary_accuracy'], label='accuracy', marker='s', color='black', linestyle='dotted', fillstyle='none')
    ax.plot(history.history['val_binary_accuracy'], label = 'val_accuracy', marker='o', color='red', linestyle='dotted', fillstyle='none')
    ax.set(xlabel='Epoch')
    ax.set(ylabel='Accuracy', ylim=[.5, 1])
    ax.legend(loc='lower right', frameon=False)
    plt.show()

if __name__ == '__main__':
    main()