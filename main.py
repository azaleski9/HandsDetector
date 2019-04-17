from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os


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


IMG_SIZE = 96

# Reizing imgages
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    #image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label


if __name__ == "__main__":

    keras = tf.keras

   # keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0,
    #                            write_graph=True, write_images=True)

    raw_dataset = tf.data.TFRecordDataset(["train.record"]).repeat()
    parsed_dataset = raw_dataset.map(_parse_function)

    all_training_elements = 182313

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

    # Own simple model - experimental - isn't used now
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    base_learning_rate = 0.0001
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    initial_epochs = 10
    steps_per_epoch = int(train_size/BATCH_SIZE)
    validation_steps = int(val_size/BATCH_SIZE)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    tb_callBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
                                              update_freq=50)

    #model = keras.models.load_model('hand_model.h5')

    # Training
    history = model.fit_generator(train_batches.repeat(),
                                  epochs=initial_epochs,
                                  workers=16, use_multiprocessing=False, max_queue_size=16,
                                  verbose=1, steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_batches.repeat(),
                                  validation_steps=validation_steps, callbacks=[cp_callback, tb_callBack])

    model.save('hand_model.h5')

    # Validating testing
    loss1, accuracy1 = model.evaluate(validation_batches, steps=validation_steps)



