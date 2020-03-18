import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt

print('Using: ')
print('TensorFlow version: {.2f}'.format(tf.__version__))
print('TensorFlow Keras version: {.2f}'.format(tf.keras.__version__))
print('Running on GPU' if tf.test.is_gpu_available() else 'GPU device not found. Running on CPU')

training_set, dataset_info = tfds.load('mnist', split='train', with_info=True, as_supervised=True)

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

num_training_examples = dataset_info.splits['train'].num_examples
# print(num_training_examples)
batch_size = 64
training_batches = training_set.cashe().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)

print(num_training_examples//4)
print(training_batches)

my_model = tf.keras.Sequential([
           tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
           tf.keras.layers.Dense(128, activation='relu'), 
           tf.keras.layers.Dense(64, activation='relu'), 
           tf.keras.layers.Dense(32, activation='relu'),
           tf.keras.layers.Dense(10, activation='softmax')
])

my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for image_batch, label_batch in training_batches.take(1):
    loss, accuracy = my_model.evaluate(image_batch, label_batch)

print('Loss before training: {:.3f}'.format(loss))
print('Accuracy before training: {:.3%}'.format(accuracy))

EPOCHS = 5
history = my_model.fit(training_batches, epochs=EPOCHS)

for image_batch, label_batch in training_batches.take(1):
    loss, accurcay = my_model.evaluate(image_batch, label_batch)

print('Loss after training: {:.3f}'.format(loss))
print('Accuracy after traingin: {:.3%}'.format(accuracy))

for image_batch, label_batch in training_batches.take(1):
    ps = my_model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]

fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(first_image, cmap = plt.cm.binary)
ax1.axis('off')
ax2.barh(np.arange(10), ps[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(np.arange(10))
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()
