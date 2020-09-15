from PIL import Image
import numpy as np
import convolutional_layer as cl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

np.random.seed(0)

# conv2d = convolutional_layer.ConvolutionLayer()

image = Image.open("./data/train/cats/cat.12.jpg")

conv2d = cl.Convolution(image, (25, 25), 3, 3, 5, 2)
print(conv2d.input.dtype)
conv2d.preprocess()
conv2d.pad()
conv2d.image.save("conv2d.jpg")
# conv2d.image.save("test_padding.jpg")
print(conv2d.filters)
res = 0
for i in range(3) :
  for j in range(3) :
    res += conv2d.input[i+6][j+4][0]
    res += conv2d.input[i+6][j+4][1]
    res += conv2d.input[i+6][j+4][2]
    
print(res)
print(conv2d.input.dtype)

np.random.seed(0)
time1 = time.time()
conv2d.convolution()
time2 = time.time()
print(time2-time1, 'seconds')

arr = np.random.choice([i for i in range(128, 256)], (200, 200, 6))
print('conv2d.feature_maps:')
print(conv2d.feature_maps.dtype)
print(conv2d.feature_maps.shape)
print(conv2d.feature_maps)
img = Image.fromarray(conv2d.feature_maps)
# print(arr)
# print(np.array(img))
# plt.imshow(arr)
# plt.show()
# img = Image.fromarray(arr)
img.save("conv.jpg")
