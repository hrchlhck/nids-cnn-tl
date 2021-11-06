from pathlib import Path
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
from tensorflow import keras

import tensorflow as tf

def main() -> None:   
    # this could also be the output a different Keras model or layer
    width, height = 800, 800
    input_tensor = Input(shape=(width, height, 3))

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


    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    data_dir = Path('data/image/2012/01/')
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(width, height),
        batch_size=64)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(width, height),
        batch_size=64)

    class_names = train_ds.class_names
    print(class_names)

    model.fit(train_ds)

    print(model.predict(val_ds))

if __name__ == '__main__':
    main()