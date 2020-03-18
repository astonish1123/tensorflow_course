import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using: ')
print('\t\u2022 Tensorflow version: ', tf.__version__)
print('\t\u2022 tf.keras version: ', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# Loading data
training_set, dataset_info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)

# Create Pipeline
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

num_training_examples = dataset_info.splits['train'].num_examples
batch_size = 64
training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)

# Building the model
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
        tf.keras.layers.Dense(128, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(10, activation='softmax')
])

# Training by using .compile method
model.compile(optimizer='adam', # The algorithm that will update the weights of the model during training
              loss='sparse_categorical_crossentropy', # The method to measure the difference between the true labels of images in dataset and the predictions made by the model
              metrics=['accuracy']) # A list of metrics to be evaluated by the model during training

for image_batch, label_batch in training_batches.take(1):
    loss, accuracy = model.evaluate(image_batch, label_batch)

print('\nLoss before training: {:.3f}'.format(loss))
print('Accuracy before training: {:.3%}'.format(accuracy))

EPOCHS = 5
history = model.fit(training_batches, epochs=EPOCHS)

for image_batch, label_batch in training_batches.take(1):
    ps = model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]

fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
ax1.imshow(first_image, cmap=plt.cm.binary)
ax1.axis('off')
ax2.barh(np.arange(10), ps[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(np.arange(10))
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()

# Evaluation of the model
for image_batch, label_batch in  training_batches.take(1):
    loss, accuracy = model.evaluate(image_batch, label_batch)

print('\nLoss after training: {:.3f}'.format(loss))
print('Accuracy after training: {:.3%}'.format(accuracy))