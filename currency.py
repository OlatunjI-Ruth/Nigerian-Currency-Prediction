from google.colab import drive
drive.mount("/content/gdrive")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers
import os
from urllib.request import urlretrieve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

training_dir = "/content/gdrive/MyDrive/Colab Notebooks/currency/Training"
val_dir = "/content/gdrive/MyDrive/Colab Notebooks/currency/Validation"

train_gen =ImageDataGenerator(rescale=1/255,
  rotation_range=40,
  width_shift_range=0.4,
  height_shift_range=0.4,
  shear_range=0.4,
  zoom_range=1,
  horizontal_flip=True,
  fill_mode='nearest')

Batch_size = 50
image_res = 400

train_data_gen=train_gen.flow_from_directory(directory=training_dir,
                                                      target_size=(image_res, image_res),
                                                      batch_size=Batch_size,
                                                      shuffle=True,
                                                      class_mode='categorical',
                                                      )

validation_gen=ImageDataGenerator(rescale=1/255)

validation_data_gen = validation_gen.flow_from_directory(directory=val_dir, 
                                                                     target_size=(image_res, image_res),
                                                                     batch_size=Batch_size, 
                                                                     class_mode='categorical',
                                                                     shuffle=False)

from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(16, (3,3), input_shape=(image_res, image_res, 3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(8, activation='softmax')
])

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

Epochs = 20

history = model.fit(train_data_gen,
                    epochs=Epochs,
                    validation_data=validation_data_gen)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(Epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

predictions = model.predict(validation_data_gen)

predictions[0]

predictions = np.argmax(predictions[0])

predictions

save_model(model, 'model1.h5')

