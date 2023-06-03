import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
from tensorflow.keras import layers

val_size = 0.2  # @param
batch_size = 32  # @param
img_height = 301  # @param
img_width = 255  # @param


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].plot(acc, label='Training Accuracy')
    axes[0].plot(val_acc, label='Validation Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training and Validation Accuracy')

    axes[1].plot(loss, label='Training Loss')
    axes[1].plot(val_loss, label='Validation Loss')
    axes[1].legend(loc='upper right')
    axes[1].set_ylabel('Cross Entropy')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('epoch')
    
    plt.show()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "img2",
  validation_split=val_size,
  subset="training",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "img2",
  validation_split=val_size,
  subset="validation",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = ["Cold","Concert[ACDC]","Gym","Casual"]


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


hidden_units = 128  # @param
dropout_rate = 0.2  # @param
learning_rate = 1e-4  # @param

mlp_model = tf.keras.Sequential([
  tf.keras.Input(shape=(img_height, img_width, 1)),
  layers.experimental.preprocessing.RandomFlip('horizontal'),
  layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
  layers.Flatten(),
  layers.Dense(hidden_units, activation='relu'),
  layers.Dropout(rate=dropout_rate),
  layers.Dense(hidden_units, activation='relu'),
  layers.Dense(4) #no of classes?
], name="MLP")

mlp_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

mlp_model.summary()

history = mlp_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)

plot_history(history)

imgwrongsize = tf.keras.utils.load_img("testimg.jpg")
imgtest = tf.image.resize(imgwrongsize,[301,255])
grayimg = tf.image.rgb_to_grayscale(imgtest)
img_array = tf.keras.utils.img_to_array(grayimg)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = mlp_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
