from PIL import Image
import numpy as np

def image_to_red_array(image) :
  result = image.copy()
  result[:, :, 1] = 0
  result[:, :, 2] = 0
  return result

def image_to_green_array(image) :
  result = image.copy()
  result[:, :, 0] = 0
  result[:, :, 2] = 0
  return result

def image_to_blue_array(image) :
  result = image.copy()
  result[:, :, 0] = 0
  result[:, :, 1] = 0
  return result

class Convolution :
  def __init__(self,
               image,
               input_size: tuple,
               n_filters: int,
               filter_dim: int,
               padding: int,
               stride: int):
    """
    Create an instance of convolution stage

    >>> image
    An instance of PIL.Image

    >>> input_size (row, col)
    Input size

    >>> n_filters
    Number of filters

    >>> filter_dim
    Dimension of filter matrix (n x n)

    >>> padding
    Padding size

    >>> stride
    Stride size
    """
    self.image = image
    self.input = np.array(self.image)
    self.input_n_rows = input_size[0]
    self.input_n_cols = input_size[1]
    self.input_depth = np.array(image).shape[2]
    self.filters = np.random.choice([-1, 0, 1], size=(filter_dim, filter_dim, self.input_depth, n_filters))
    # self.filters = np.full((n_filters, self.input_depth, filter_dim, filter_dim), 1, self.input.dtype)
    self.biases = np.zeros(n_filters)
    self.padding = padding
    self.stride = stride

    self.feature_maps = np.zeros((int(((self.input_n_rows - filter_dim + 2 * self.padding) / self.stride) + 1),
                                  int(((self.input_n_cols - filter_dim + 2 * self.padding) / self.stride) + 1),
                                  n_filters),
                                  self.input.dtype)

  def preprocess(self) :
    """
    Resizing and cropping the image based on the input size
    """
    self.image.thumbnail((self.input_n_cols, self.input_n_rows))

    image_n_cols, image_n_rows = self.image.size
    left = (image_n_cols - self.input_n_cols) / 2
    top = (image_n_rows - self.input_n_rows) / 2
    right = (image_n_cols + self.input_n_cols) / 2
    bottom = (image_n_rows + self.input_n_rows) / 2

    self.image = self.image.crop((left, top, right, bottom))
    self.input = np.array(self.image)
    self.input_n_rows, self.input_n_cols, self.input_depth = self.input.shape

  def pad(self) :
    """
    Add zero-value padding around the input matrix based on the padding size
    """
    left_input = top_input = self.padding
    right_input, bottom_input = self.input.shape[1] + self.padding, self.input.shape[0] + self.padding
    padded_input_rows, padded_input_cols = self.input.shape[0] + 2 * self.padding, self.input.shape[1] + 2 * self.padding
    depth = self.input_depth
    
    result = np.zeros_like(self.input, shape=(padded_input_rows, padded_input_cols, depth))
    result[top_input:bottom_input, left_input:right_input] = self.input

    self.image = Image.fromarray(result)
    self.input = np.array(self.image)
    self.input_n_rows, self.input_n_cols, self.input_depth = self.input.shape

  def convolution(self) :
    # for current_filter in range(len(self.filters)) :
    #   print('current_filter:', current_filter)
    #   result = 0
    #   for feature_map_row in range(len(self.feature_maps)) :
    #     print('  feature_map_row:', feature_map_row)
    #     for feature_map_col in range(len(self.feature_maps[feature_map_row])) :
    #       print('    feature_map_col:', feature_map_col)
    #       for depth in range(self.input_depth) :
    #         print('      depth:', depth)
    #         for filter_row in range(0, len(self.filters[current_filter][depth])) :
    #           print('        filter_row:', filter_row)
    #           for filter_col in range(0, len(self.filters[current_filter][depth])) :
    #             print('          filter_col:', filter_col)
    #             result += self.input[feature_map_row * self.stride + filter_row][feature_map_col * self.stride + filter_col][depth] * self.filters[current_filter][depth][filter_row][filter_col]
    #             # print(self.input[feature_map_row * self.stride + filter_row][feature_map_col * self.stride + filter_col][depth], self.filters[current_filter][depth][filter_row][filter_col])
    #       result += self.biases[current_filter]
    #       self.feature_maps[feature_map_row][feature_map_col][current_filter] = int(result)
    pass