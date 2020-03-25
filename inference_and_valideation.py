import warnings
warnings.filterwarnings("igonre")
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using: ')
print('TensorFlow version: {}'.formta(tf.__version__))
print('tf.keras version: {}'.format(tf.keras.__version__))
print('Running on GPU' if tf.test.is_gpu_available() else 'GPU device not found. Running on CPU')

# train_split = 60
# test_val_split = 20
# splits = tfds.Split.ALL.subsplit([train_split, test_val_split, test_val_split])
dataset, dataset_info = tfds.load('fashion_mnist', split=['train[:50000]', 'test', 'train[50000:]'], as_supervised=True, with_info=True)
training_set, validation_set, test_set = dataset

print('dataset has type: ', type(dataset))
print('dataset has {:} elements '.format(len(dataset)))
print(dataset)
print(dataset_info)

total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples
num_training_examples = (total_examples * 60) // 100
num_validation_examples = (total_examples * 20) // 100
num_test_examples = num_validation_examples

print('There are {} images in the training set'.format(num_training_examples))
print('There are {} images in the validation set'.format(num_validation_examples))
print('There are {} images in the test set'.format(num_test_examples))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64
training_batch = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batch = validation_set.cache().batch(batch_size).map(normalize).prefetch(1)
testing_batch = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(256, activation='relu'),  
        tf.keras.layers.Dense(128, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

loss, accuracy = model.evaluate(testing_batch)
print('Loss on the TEST set: {:.3f}'.format(loss))
print('Accuracy on the TEST set: {:.3%}'.format(accuracy))

EPOCHS = 30
HISTORY = model.fit(training_batch, epochs=EPOCHS, validation_data=validation_batch)

print('HISTORY.history has type: ', type(HISTORY.history))
print('The keys of HISTORY.hisotry are: ', list(HISTORY.history.keys()))

training_accuracy = HISTORY.history['accuracy']
validation_accuracy = HISTORY.history['val_accuracy']

training_loss = HISTORY.history['loss']
validation_loss = HISTORY.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_2 = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
          tf.keras.layers.Dense(254, activation='relu'), 
          tf.keras.layers.Dense(228, activation='relu'),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
])
model_2.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
# Stop training when there is no improvement in the validation loss for 5 cnsecutive epochs
early_stoppoing = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

HISTORY_2 = model_2.fit(training_batch, epochs=100, validation_data=validation_batch, callbacks=[early_stoppoing])

training_accuracy_2 = HISTORY_2.history('accuracy')
validation_accuracy_2 = HISTORY_2.history('val_accuracy')
training_loss_2 = HISTORY_2.history('loss')
validation_loss_2 = HISTORY_2.hitory('val_loss')

epochs_range = range(len(training_accuracy))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy_2, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy_2, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accurcay')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss_2, label='Training Loss')
plt.plot(epochs_range, validation_loss_2, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model_3 = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
          tf.keras.layers.Dense(254, activation='relu'), 
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation='softmax')
])
model_3.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy')

HISTORY_3 = model_3.fit(training_batch, epochs=EPOCHS, validation_data=validation_batch)

training_accuracy_3 = HISTORY_3.history['accuracy']
validation_accuracy_3 = HISTORY_3.history['val_accuracy']
training_loss_3 = HISTORY_3.history('loss')
validation_loss_3 = HISTORY_3.history('val_loss')

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy_3, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy_3, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss_3, label='Training Loss')
plt.plot(epochs_range, validation_loss_3, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

for image_batch, label_batch in testing_batch.take(1):
    ps = model_3.predict(image_batch)
    images = image_batch.numpy().squeeze()
    labels = label_batch.numpy()

plt.figure(figsize=(10, 15))
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.imshow(images[n], cmap=plt.cm.binary)
    color='green' if np.argmax(ps[n]) == labels[n] else 'red'
    plt.title(class_names[np.argmax(ps[n])], color=color)
    plt.axis('off')