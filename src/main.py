import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score

from model.sequential import SequentialModel
from model.layers.layer import Layer
from model.layers.convolutional import Convolutional
from model.layers.dense import Dense
from model.layers.flatten import Flatten
from model.layers.pooling import Pooling

np.random.seed(13517)

def load_images_as_dataset(directory, image_size, batch_size, rescale=True) :
  test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=['dogs', 'cats'],
    batch_size=batch_size,
    image_size=image_size
  )
  rescale_factor = 1.0/255 if rescale else 1

  list_images = []
  list_labels = []
  for images, labels in test_dataset.take(1) :
    for i in range(len(images)) :
      list_images.append(images[i].numpy().transpose(2, 0, 1) * rescale_factor)
      list_labels.append(labels[i].numpy())
  return list_images, list_labels

if __name__ == "__main__" :
  IMG_DIR_TEST = '../data/test'
  IMG_DIR_TRAINING = '../data/train'
  IMAGE_SIZE = (150, 150)
  BATCH_SIZE = 15

  # Prepare dataset
  list_test_images, list_test_labels = load_images_as_dataset(IMG_DIR_TEST, IMAGE_SIZE, BATCH_SIZE)
  list_train_images, list_train_labels = load_images_as_dataset(IMG_DIR_TRAINING, IMAGE_SIZE, BATCH_SIZE)

  # Define models
  model1 = SequentialModel([
    Convolutional(4, (3, 3), (150, 150, 3), 0, 1),
    Pooling((2, 2), 1),
    Convolutional(8, (3, 3)),
    Pooling((2, 2), 1),
    Flatten(),
    Dense(256, 'relu'),
    Dense(1, 'sigmoid')
  ])
  print(list_test_labels)
  # # model2 = SequentialModel([
  # #   Convolutional(16, (3, 3), (150, 150, 3)),
  # #   Pooling((2, 2), 2),
  # #   Convolutional(32, (3, 3)),
  # #   Pooling((2, 2), 2),
  # #   Convolutional(64, (3, 3)),
  # #   Pooling((2, 2), 2),
  # #   Flatten(),
  # #   Dense(512, 'relu'),
  # #   Dense(1, 'sigmoid')
  # # ])

  # # Load model
  # # model1.load_model_from_json('testing.json')

  # # List of predicted labels by model
  list_predicted = []

  # model1.fit2(list_images[1:2], list_labels[1:2])
  model1.fit(list_train_images, list_train_labels, batch_size=5)

  # # Predict using defined models
  print('\n================')
  print('Predict')
  print('================')
  list_predicted = model1.predict(np.array(list_test_images), np.array(list_test_labels), True)
  print('Model accuracy:', accuracy_score(list_test_labels, list_predicted))

  # Save model
  model1.save_model_as_json('model1.json')