from PIL import Image
import numpy as np
import time

from model.sequential import SequentialModel
from model.layers.layer import Layer
from model.layers.convolutional import Convolutional
from model.layers.dense import Dense
from model.layers.flatten import Flatten
from model.layers.pooling import Pooling

np.random.seed(0)

IMG_DIR_TEST = './data/test'
IMAGE_SIZE = (150, 150)

test = tf.keras.preprocessing.image_dataset_from_directory(
  IMG_DIR_TEST,
  image_size=IMAGE_SIZE
)

model = SequentialModel([
  Convolutional(4, (3, 3), (150, 150, 3), 0, 1),
  Pooling((2, 2), 2),
  Convolutional(8, (3, 3)),
  Pooling((2, 2), 2),
  Flatten(),
  Dense(256, 'relu'),
  Dense(1, 'sigmoid')
])
image = Image.open("./data/test/cats/cat.15.jpg")
dummy_array = np.array([[[85, 170, 255],
                         [170, 255, 85],
                         [255, 85, 170]],
                        
                        [[170, 255, 85],
                         [255, 85, 170],
                         [85, 170, 255]],
                         
                        [[255, 85, 170],
                         [85, 170, 255],
                         [170, 255, 85]],
                         
                        [[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]])
# print(dummy_array)
# model.forward(image)
print(model.output_shapes)
# print(model.forward(arr))