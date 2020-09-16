from PIL import Image
import numpy as np
import convolutional_layer as cl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2

def resize_col(array) :
  pass

np.random.seed(0)

# conv2d = convolutional_layer.ConvolutionLayer()

image = Image.open("./data/train/cats/cat.12.jpg")
array = np.array(image)

# conv2d_img = cl.Convolution(image, (300, 300), 3, 3, 5, 2)
# conv2d_arr = cl.Convolution(array, (300, 300), 3, 3, 5, 2)
dummy_array = np.array([[[85, 170, 255], [170, 255, 85], [255, 85, 170]],
                        [[170, 255, 85], [255, 85, 170], [85, 170, 255]],
                        [[255, 85, 170], [85, 170, 255], [170, 255, 85]]])
conv2d = cl.Convolution(dummy_array, (2, 2), 3, 2, 1, 1)
conv2d.preprocess()
print(conv2d.input)
print(conv2d.input.shape)
conv2d.convolution()
print(conv2d.input)
print(conv2d.input.shape)
print(conv2d.feature_maps)
print(conv2d.feature_maps.shape)
# print(conv2d_img.input.shape)
# print(conv2d_arr.input.shape)

# conv2d_img.resize()
# conv2d_arr.resize()
# conv2d_img.crop()
# conv2d_arr.crop()
# conv2d_img.preprocess()
# conv2d_arr.preprocess()


# print(conv2d_img.input.shape)
# print(conv2d_arr.input.shape)

# conv2d_img.image.save("conv2d_img.jpg")
# Image.fromarray(conv2d_arr.input).save("conv2d_arr.jpg")
# Image.fromarray(conv2d.feature_maps, 'RGB').save("dummy.jpg")