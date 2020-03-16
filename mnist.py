import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using: ')
print('\t\u2022 TensorFlow version: ', tf.__version__)
print('\t\u2022 tf.keras version: ', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# Data loading from the TensorFlow Dataset into training_set and dataset_info
training_set, dataset_info = tfds.load('mnist', split='train', as_supervised=True, with_info=True)
num_classes = dataset_info.features['label'].num_classes
print('There are {:,} classes in our dataset'.format(num_classes))
num_training_examples = dataset_info.splits['train'].num_examples
print('\nThere are {:,} images in the training set'.format(num_training_examples))

for image, label in training_set.take(1):
    print('The images in the training set have: ')
    print('\u2022 dtype: ', image.dtype)
    print('\u2022 shape: ', image.shape)
    print('\nThe labels of the images have: ')
    print('\u2022 dtype: ', label.dtype)

# Take one image from training_set and cast the type to numpy and squeeze in 2D
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar() # I can see the pixel values of the image
plt.show()
print('The label of this image is: ', label)

# In order to normalize image, cast type of the image into float32 and devide it by the pixel value, 255
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)

for image_batch, label_batch in training_batches.take(1):
    print('The images in each batch have: ')
    print('\u2022 dtype: ', image_batch.dtype)
    print('\u2022 shape: ', image_batch.shape)
    print('\nThere are a total of {} image labels in this batch: ', format(label_batch.numpy().size))
    print(label_batch.numpy())

for image_batch, label_batch in training_batches.take(1):
    images = image_batch.numpy().squeeze()
    labels = label_batch.numpy()

plt.imshow(images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is: ', labels[0])

# activation function Sigomoid function
def activation(x):
    return 1/(1 + tf.exp(-x))

# Flatten the input images because inputs to every layer must be one-dimensional vector
inputs = tf.reshape(images, [images.shape[0], -1])
# Print the shape of the inputs, (64, 784) 784 = 28 * 28 
print('The inputs have shape: ', inputs.shape)
# Initiate Neural Network parameters for weights and bias
w1 = tf.random.normal((784, 256))
b1 = tf.random.normal((1, 256))

w2 = tf.random.normal((256, 10))
b2 = tf.random.normal((1, 10))
# Perform matricx multiplaications for the hidden layer
# and apply activation fucntion
h = activation(tf.matmul(inputs, w1) + b1)
# Perform matrix multiplication for the output layer
output = tf.matmul(h, w2) + b2
# Print the shape of the output. It should be (64, 10)
print('The output has shape: ', output.shape)

def softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)

probabilities = softmax(output)

# Print the shape of the probabilites. (64, 10)
print('The probabilities have shape: ', probabilities.shape, '\n')
# The sum of probabilities for each of the 64 images should be 1
sum_all_prob = tf.reduce_sum(probabilities, axis=1).numpy()

# Print the sum of the probabiities for each image.
for i, prob_sum in enumerate(sum_all_prob):
    print('Sum of probabilities for Image {}: {:.1f}'.format(i+1, prob_sum))

# Develop keras model
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
        tf.keras.layers.Dense(256, activation = 'sigmoid'),
        tf.keras.layers.Dense(10, activation= 'softmax')
])
model.summary()

my_model_1 = tf.keras.Sequential([
             tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
             tf.keras.layers.Dense(128, activation='relu'),
             tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(10, activation='softmax')
])
my_model_1.summary()

model_weights_biases = model.get_weights()
print(type(model_weights_biases))
print('\nThere are {:,} Numpy ndarrays in our list\n'.format(len(model_weights_biases)))
print(model_weights_biases)

model.layers

for i, layer in enumerate(model.layers):
    
    if len(layer.get_weights()) > 0:
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]
        
        print('\nLayer {}: {}\n'.format(i, layer.name))
        print('\u2022 Weights:\n', w)
        print('\n\u2022 Biases:\n', b)
        print('\nThis layer has a total of {:,} weights and {:,} biases'.format(w.size, b.size))
        print('\n------------------------')
    
    else:
        print('\nLayer {}: {}\n'.format(i, layer.name))
        print('This layer has no weights or biases.')
        print('\n------------------------')

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