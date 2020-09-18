from PIL import Image
import numpy as np
import convolutional as conv
import pooling as pool

np.random.seed(0)

# conv2d = convolutional_layer.ConvolutionLayer()

# ## input from image
# image = Image.open("./data/train/cats/cat.12.jpg")
# conv2d_img = conv.Convolutional(image, (300, 300), 3, 3, 5, 2)
# print('conv2d_img.input.shape:', conv2d_img.input.shape)
# conv2d_img.preprocess()
# print('conv2d_img.input.shape:', conv2d_img.input.shape)
# conv2d_img.convolution()
# conv2d_img.detector()
# Image.fromarray(conv2d_img.feature_maps.astype('uint8')).save('img.jpg')
# print('conv2d_img.feature_maps:')
# print(conv2d_img.feature_maps)
# print(conv2d_img.feature_maps.shape)
# pool2d_img = pool.Pooling(conv2d_img.feature_maps)
# pool2d_img.pooling()
# Image.fromarray(pool2d_img.feature_maps.astype('uint8')).save('img_pool.jpg')
# print('pool2d_img.feature_maps:')
# print(pool2d_img.feature_maps)
# print('pool2d_img.feature_maps.shape:', pool2d_img.feature_maps.shape)

### input from array from image
# array = np.array(image)
# conv2d_arr = conv.ConvolutionalLayer(array, (300, 300), 3, 3, 5, 2)
# print('conv2d_arr.input.shape:', conv2d_arr.input.shape)
# conv2d_arr.preprocess()
# print('conv2d_arr.input.shape:', conv2d_arr.input.shape)
# conv2d_arr.convolution()
# conv2d_arr.detector()
# print('conv2d_arr.feature_maps:')
# print(conv2d_arr.feature_maps)
# print(conv2d_arr.feature_maps.shape)
# pool2d_arr = pool.Pooling(conv2d_arr.feature_maps)
# pool2d_arr.pooling()
# print('pool2d_arr.feature_maps:')
# print(pool2d_arr.feature_maps)
# print('pool2d_arr.feature_maps.shape:', pool2d_arr.feature_maps.shape)

## input from array (manual)
dummy_array = np.array([[[85, 170, 255], [170, 255, 85], [255, 85, 170]],
                        [[170, 255, 85], [255, 85, 170], [85, 170, 255]],
                        [[255, 85, 170], [85, 170, 255], [170, 255, 85]]])
conv2d = conv.Convolutional(dummy_array, (2, 2), 5, 2, 1, 1)
print('conv2d.input.shape:', conv2d.input.shape)
conv2d.preprocess()
print('conv2d.input.shape:', conv2d.input.shape)
conv2d.convolution()
conv2d.detector()
print('conv2d.feature_maps:')
print(conv2d.feature_maps)
print('conv2d.feature_maps.shape:', conv2d.feature_maps.shape)
pool2d = pool.Pooling(conv2d.feature_maps, stride=1)
pool2d.pooling()
print('pool2d.feature_maps:')
print(pool2d.feature_maps)
print('pool2d.feature_maps.shape:', pool2d.feature_maps.shape)

# Image.fromarray(conv2d_img.feature_maps, 'RGB').save("conv2d_img.jpg")
# Image.fromarray(conv2d_arr.feature_maps, 'RGB').save("conv2d_arr.jpg")
# Image.fromarray(conv2d.feature_maps, 'RGB').save("dummy.jpg")