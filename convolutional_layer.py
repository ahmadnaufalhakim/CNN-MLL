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

class ConvolutionalLayer :
  def __init__(self,
               input,
               input_size: tuple,
               n_filters: int,
               filter_dim: int,
               padding: int = None,
               stride: int = None) :
    """
    Create an instance of convolution layer

    >>> input
    An instance of PIL.Image or a Numpy Array

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
    if Image.isImageType(input) :
      self.image = input
      self.input = np.array(self.image).astype("float")
    else :
      self.input = input.astype("float")

    self.input_n_rows = input_size[0]
    self.input_n_cols = input_size[1]
    self.input_depth = self.input.shape[2]
    self.filters = np.random.choice([-1, 0, 1], size=(n_filters, filter_dim, filter_dim, self.input_depth)).astype("float")
    # self.filters = np.full((filter_dim, filter_dim, self.input_depth, n_filters), 1, self.input.dtype)
    self.biases = np.zeros(n_filters)
    self.padding = padding if padding is not None else 0
    self.stride = stride if stride is not None else 1
    self.feature_maps = np.zeros((int(((self.input_n_rows - filter_dim + 2 * self.padding) / self.stride) + 1),
                                  int(((self.input_n_cols - filter_dim + 2 * self.padding) / self.stride) + 1),
                                  n_filters),
                                  self.input.dtype)

  def resize(self) :
    """
    If attribute input is an instance of PIL.Image, then use method from PIL library to resize the image
    """
    if hasattr(self, "image") :
      self.image.thumbnail((self.input_n_cols, self.input_n_rows))
      self.input = np.array(self.image)
    else :
      # TODO: Resize input matrix manually and maintain visual aspect
      pass

  def crop(self) :
    """
    If self has an image attribute, then crop using the method from PIL library, else crop input matrix manually
    """
    if hasattr(self, "image") :
      image_n_cols, image_n_rows = self.image.size
      left = (image_n_cols - self.input_n_cols) // 2
      top = (image_n_rows - self.input_n_rows) // 2
      right = (image_n_cols + self.input_n_cols) // 2
      bottom = (image_n_rows + self.input_n_rows) // 2
      self.image = self.image.crop((left, top, right, bottom))
      self.input = np.array(self.image)
    else :
      image_n_rows, image_n_cols, _ = self.input.shape
      input_left = (image_n_cols - self.input_n_cols) // 2
      input_top = (image_n_rows - self.input_n_rows) // 2
      input_right = (image_n_cols + self.input_n_cols) // 2
      input_bottom = (image_n_rows + self.input_n_rows) // 2

      rows = max((input_bottom - input_top), image_n_rows)
      cols = max((input_right - input_left), image_n_cols)
      res = np.zeros_like(self.input, shape=(rows, cols, self.input_depth))
      res_left, res_top, res_right, res_bottom = input_left, input_top, input_right, input_bottom

      if rows == image_n_rows :
        res_top = 0
        res_bottom = image_n_rows
      elif rows == input_bottom - input_top :
        res_top = -res_top
        input_top = 0
        input_bottom = rows
      if cols == image_n_cols :
        res_left = 0
        res_right = image_n_cols
      elif cols == input_right - input_left :
        res_left = -res_left
        input_left = 0
        input_right = cols

      res[res_top:res_bottom, res_left:res_right] = self.input
      self.input = res[input_top:input_bottom, input_left:input_right]

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

    if hasattr(self, "image") :
      self.image = Image.fromarray(result)

    self.input = result

  def preprocess(self) :
    """
    Resizing, cropping, and add padding to the image based on the input size
    """
    self.resize()
    self.crop()
    self.pad()

  def convolution(self) :
    """
    Convolute input matrix with filters/kernels
    """
    for feature_map_row in range(len(self.feature_maps)) :
      for feature_map_col in range(len(self.feature_maps[feature_map_row])) :
        for feature_map_depth in range(len(self.feature_maps[feature_map_row][feature_map_col])) :
          result = 0
          for filter_row in range(self.filters.shape[1]) :
            for filter_col in range(self.filters.shape[2]) :
              for filter_depth in range(self.filters.shape[3]) :
                result += self.input[feature_map_row * self.stride + filter_row][feature_map_col * self.stride + filter_col][filter_depth] * self.filters[feature_map_depth][filter_row][filter_col][filter_depth]
          result += self.biases[feature_map_depth]
          self.feature_maps[feature_map_row][feature_map_col][feature_map_depth] = result

  def detector(self) :
    """
    Apply ReLU activation function to the feature maps
    """
    for feature_map_row in range(len(self.feature_maps)) :
      for feature_map_col in range(len(self.feature_maps[feature_map_row])) :
        for feature_map_depth in range(len(self.feature_maps[feature_map_row][feature_map_col])) :
          if self.feature_maps[feature_map_row][feature_map_col][feature_map_depth] < 0 :
            self.feature_maps[feature_map_row][feature_map_col][feature_map_depth] = 0