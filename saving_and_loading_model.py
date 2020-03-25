import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Uing: ')
print('TensorFlow Version: ', tf.__version__)
print('tf.keras version: ', tf.keras.__version__)
print('Running on GPU' if tf.test.is_gpu_available() else 'GPU device not found. Running on CPU')

# Load Dataset
dataset, dataset_info = tfds.load('fashion_mnist', split=['train[:50000]', 'test', 'train[50000:]'], as_supervised=True, with_info=True)
training_set, validation_set, test_set = dataset

# Explore Dataset
total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples
print(total_examples)
num_training_examples = (total_examples * 60) // 100
print(num_training_examples)
num_validation_examples = (total_examples * 20) //100
print(num_validation_examples)
num_test_examples = num_validation_examples
print(num_test_examples)
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()
plt.imshow(image, cmap=plt.cm.binary)
plt.title(class_name[label])
plt.colorbar()
plt.show()

# Create Pipeline
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label
batch_size = 64
training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batches = validation_set.cache().batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

# Build and train model
layer_neurons = [512, 256, 128]
dropout_rate = 0.5
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 4
history = model.fit(training_batches, epochs=EPOCHS, validation_data=validation_batches)

# Saving and Loading model in HDF5 format (keras)
t = time.time()
saved_keras_model_filepath = './{}.h5'.format(int(t))
model.save(saved_keras_model_filepath)

reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath)
reloaded_keras_model.summary()

for image_batch, label_batch in testing_batches.take(1):
    prediction_1 = model.predict(image_batch)
    prediction_2 = reloaded_keras_model.predict(image_batch)
    difference = np.abs(prediction_1 - prediction_2)
    print(difference.max())

# Saving and Loading TensorFlow SavedModels
t = time.time()
savedModel_directory = './{}'.format(int(t))
tf.saved_model.save(model, savedModel_directory)
reloaded_SavedModel = tf.saved_model.load(savedModel_directory)
for image_batch, label_batch in testing_batches.take(1):
    prediction_1 = model.predict(image_batch)
    prediction_2 = reloaded_SavedModel(image_batch, training=False).numpy()
    difference = np.abs(prediction_1 - prediction_2)
    print(difference.max())

# Convert back savedModle into keras model
reloaded_keras_model_from_SavedModel = tf.keras.models.load_model(savedModel_directory)
reloaded_keras_model_from_SavedModel.summary()

# Saving models during training
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(training_batches, epochs=100, validation_data=validation_batches, callbacks=[early_stopping, save_best])
