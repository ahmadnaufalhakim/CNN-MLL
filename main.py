from PIL import Image
import numpy as np
from glob import glob
import time
# import tensorflow as tf

from model.sequential import SequentialModel
from model.layers.layer import Layer
from model.layers.convolutional import Convolutional
from model.layers.dense import Dense
from model.layers.flatten import Flatten
from model.layers.pooling import Pooling

np.random.seed(0)

# IMG_DIR_TEST = './data/test'
# IMAGE_SIZE = (150, 150)

# test = tf.keras.preprocessing.image_dataset_from_directory(
#   IMG_DIR_TEST,
#   image_size=IMAGE_SIZE
# )

model1 = SequentialModel([
  Convolutional(4, (3, 3), (150, 150, 3), 0, 1),
  Pooling((2, 2), 2),
  Convolutional(8, (3, 3)),
  Pooling((2, 2), 2),
  Flatten(),
  Dense(256, 'relu'),
  Dense(1, 'sigmoid')
])

# model2 = SequentialModel([
#   Convolutional(4, (3, 3), (150, 150, 3), 0, 1),
#   Pooling((2, 2), 2),
#   Convolutional(8, (3, 3)),
#   Pooling((2, 2), 2),
#   Flatten(),
#   Dense(256, 'relu'),
#   Dense(1, 'relu')
# ])

images = glob('./data/test/cats/*.jpg')
images.extend(glob('./data/test/dogs/*.jpg'))

# dummy_array = np.array([[[85, 170, 255],
#                          [170, 255, 85],
#                          [255, 85, 170]],
                        
#                         [[170, 255, 85],
#                          [255, 85, 170],
#                          [85, 170, 255]],
                         
#                         [[255, 85, 170],
#                          [85, 170, 255],
#                          [170, 255, 85]],
                         
#                         [[1, 2, 3],
#                          [4, 5, 6],
#                          [7, 8, 9]]])

for img in images :
  print('\nfilename:', img)
  arr = np.array(Image.open(img)).transpose(2, 0, 1) * (1.0/255)

  start1 = time.time()
  print('model1:', model1.forward(Image.open(img)))
  finish1 = time.time()
  print('model1 finished in:', finish1-start1)
  
  # start2 = time.time()
  # print('model2:', model2.forward(Image.open(img)))
  # finish2 = time.time()
  # print('model2 finished in:', finish2-start2)