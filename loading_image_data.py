import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using: ')
print('TensorFlow Version: ', tf.__version__)
print('TensorFlow Keras Version: ', tf.keras.__version__)
print('Running on GPU' if tf.test.is_gpu_available() else 'GPU devise not found. Running on CPU')

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('The dataset contains: ')
print('{:} training images'.format(total_train))
print('{:} validation images'.format(total_val))
print('The training set contains: ')
print('{:} cat images'.format(num_cats_tr))
print('{:} dog images'.format(num_dogs_tr))
print('The validation set contains: ')
print('{:} cat images'.format(num_cats_val))
print('{:} dog images'.format(num_dogs_val))

BATCH_SIZE = 64
IMG_SHAPE = 224
# image_gen = ImageDataGenerator(rescale=1./255)
# one_image = image_gen.flow_from_directory(directory=train_dir, batch_size=BATCH_SIZE, shuffle=True, 
#                                           target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

# plt.imshow(one_image[0][0][0])
# plt.show()

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
# Flipping Images Horizontally
# image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
# train_data_gen = image_gen.flow_from_directory(directory=train_dir, 
#                                                batch_size=BATCH_SIZE,
#                                                shuffle=True,
#                                                target_size=(IMG_SHAPE, IMG_SHAPE),
#                                                class_mode='binary')

# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# # Rotating the image
# image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
# train_data_gen = image_gen.flow_from_directory(directory=train_dir, batch_size=BATCH_SIZE,
#                                                shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE),
#                                                class_mode='binary')
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# Applying Flipping, Rotating, and zoom
image_gen_train = ImageDataGenerator(rescale=1./255, horizontal_flip=True, 
                               rotation_range=45, zoom_range=0.5,
                               width_shift_range=0.2, 
                               height_shift_range=0.2, 
                               shear_range=0.2,
                               fill_mode='nearest')
train_data_gen = image_gen_train.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(directory=validation_dir, batch_size=BATCH_SIZE,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE), 
                                                 class_mode='binary')

layer_neurons = [1024, 512, 256, 128, 56, 28, 14]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(IMG_SHAPE, IMG_SHAPE, 3)))
for neuron in layer_neurons:
    model.add(tf.keras.layers.Dense(neuron, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 10
model.fit(train_data_gen, epochs=EPOCHS, validation_data=val_data_gen)