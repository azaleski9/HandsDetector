from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.python.data.ops import dataset_ops


#TFRecords parser
def _parse_function(example_proto):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)
        })
    #image = tf.io.decode_raw(features['image/encoded'], tf.float32)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/object/class/label'], tf.int64)
    return image, label


IMG_SIZE = 160 # All images will be resized to 160x160


# Reizing imgages to 160x160 pixels
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label


if __name__ == "__main__":

    keras = tf.keras

    raw_dataset = tf.data.TFRecordDataset(["train.record"]).repeat()
    parsed_dataset = raw_dataset.map(_parse_function)

    all_training_elements = 4400
    train_size = int(0.85 * all_training_elements)
    val_size = int(0.15 * all_training_elements)

    raw_dataset_test = tf.data.TFRecordDataset(["test.record"]).repeat()
    parsed_dataset_test = raw_dataset.map(_parse_function)

    raw_test = parsed_dataset_test
    raw_validation = parsed_dataset.take(val_size)
    raw_train = parsed_dataset

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    for image_batch, label_batch in train_batches.take(1):
        pass

    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    prediction_layer = keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)

    # Own simple model - experimental - isn't used now
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Premade model from Tensorflow docs
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #training
   # num_train, num_val, num_test = (
    #    metadata.splits['train'].num_examples * weight / 10
     #   for weight in SPLIT_WEIGHTS
    #)

    initial_epochs = 3
    steps_per_epoch = int(train_size/BATCH_SIZE)
    validation_steps = int(val_size/BATCH_SIZE)

    # Training
    history = model.fit(train_batches.repeat(),
                        epochs=initial_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_batches.repeat(),
                        validation_steps=validation_steps)

    # Validating testing
    loss1, accuracy1 = model.evaluate(validation_batches, steps=validation_steps)



