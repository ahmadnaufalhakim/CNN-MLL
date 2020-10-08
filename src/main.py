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
  IMAGE_SIZE = (150, 150)
  BATCH_SIZE = 40

  # Prepare dataset
  list_images, list_labels = load_images_as_dataset(IMG_DIR_TEST, IMAGE_SIZE, BATCH_SIZE)

  # Define models
  model1 = SequentialModel([
    Convolutional(4, (3, 3), (150, 150, 3), 0, 1),
    Pooling((2, 2), 2),
    Convolutional(8, (3, 3)),
    Pooling((2, 2), 2),
    Flatten(),
    Dense(256, 'relu'),
    Dense(1, 'sigmoid')
  ])

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
  # print(model1.output_shapes)
  # # List of predicted labels by model
  # list_predicted = []

  # model1.fit2(list_images[1:2], list_labels[1:2])
  model1.fit(list_images[1:6], list_labels[1:6])

  # # Predict using defined models
  print('\n================')
  print('Predict')
  print('================')
  # list_predicted = model1.predict(np.array(list_images[0:1]), np.array(list_labels[0:1]), True)
  # print(list_predicted.shape)
  # print('Model accuracy:', accuracy_score(list_labels[0:1], list_predicted[0:1]))

  # Save model
  # model1.save_model_as_json('testing.json')

  # print(np.dot(np.array([[9.855923e-1],
  #                        [1.419998e-2],
  #                        [2.045877e-4],
  #                        [2.947614e-6],
  #                        [4.246805e-8],
  #                        [6.118620e-10],
  #                        [8.815459e-12],
  #                        [6.118620e-10],
  #                        [4.246805e-8],
  #                        [-1]]),
  #              np.array([[1, 424, 0]])))
  # a = np.array([1,2,3,4,5,6,7,8,9,10])
  # print(a.reshape(10, 1))
  # print(a, a.shape)

  # # TESTING DENSE BACKWARD
  # dns = Dense(10, 'relu', 2)
  # w = np.array([[0.09,0.02],
  #               [0.08,0.03],
  #               [0.07,0.03],
  #               [0.06,0.02],
  #               [0.05,0.01],
  #               [0.04,0.02],
  #               [0.03,0.07],
  #               [0.04,0.08],
  #               [0.05,0.05],
  #               [0.01,0.01]])
  # dns.weights = w
  # print(dns.biases)
  # print(dns.weights)
  # print(dns.delta_weights)

  # error = np.array([[9.855923e-1],
  #                   [1.419998e-2],
  #                   [2.045877e-4],
  #                   [2.947614e-6],
  #                   [4.246805e-8],
  #                   [6.118620e-10],
  #                   [8.815459e-12],
  #                   [6.118620e-10],
  #                   [4.246805e-8],
  #                   [-1]])
  # print(error[3])

  # dns.input = np.array([424, 0])
  # print(dns.derivative_weight(error))
  # print(dns.derivative_input(error))
  # print(dns.backward(error, 1, 1))
  # print(dns.delta_weights)
  # dns.update_weights()
  # print(dns.biases)
  # print(dns.weights)
  # print(dns.delta_weights)

  # # TESTING FLATTEN BACKWARD
  # flat = Flatten()
  # flat.input_shape = (2,3,3)
  # err = np.array([n for n in range(1,19)])
  # print(flat.backward(err))

  # print(w.shape)
  # print(w.transpose())
  # print(np.zeros(5).transpose().shape)

  # a = np.zeros(3)
  # a[0] = 1
  # a[1:] = [424,0]
  # print(a.reshape(3,1))
  # print(w[:, 1:])