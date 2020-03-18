import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using: ')
print('TensorFlow version: ', tf.__version__)
print('tf.keras version: ', tf.keras.__version__)
print('Running on GPU' if tf.test.is_gpu_available() else 'GPU device not found. Running on CPU')

dataset, dataset_info = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
print('dataset has type: ', type(dataset))
print('The keys of dataset are: ', list(dataset.keys()))

training_set, test_set = dataset['train'], dataset['test']
print(dataset_info)

print(dataset_info.features['image'])
print(dataset_info.features['label'])
print(dataset_info.splits['train'])
shape_images = dataset_info.features['image'].shape
num_classes = dataset_info.features['label'].num_classes
num_training_examples = dataset_info.splits['train'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

print('There are {:} classes in our dataset'.format(num_classes))
print('The images in our dataset have shape: ', shape_images)
print('There are {:} images in the test set'.format(num_test_examples))
print('There are {:} images in the training set'.format(num_training_examples))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for image, label in training_set.take(1):
    print('The images in the training set have: \ndtype:', image.dtype, '\nshpae:', image.shape)

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is: ', label)
print('The class name of this image is: ', class_names[label])

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64
training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCH = 5
model.fit(training_batches, epochs=EPOCH)

loss, accuracy = model.evaluate(testing_batches)
print('Loss on the TEST set: {:.3f}'.format(loss))
print('Accuracy on the TEST set: {:.3%}'.format(accuracy))

for image_batch, label_batch in testing_batches.take(1):
    ps = model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]
    first_label = label_batch.numpy()[0]

fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(first_image, cmap = plt.cm.binary)
ax1.axis('off')
ax1.set_title(class_names[first_label])
ax2.barh(np.arange(10), ps[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(class_names, size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()