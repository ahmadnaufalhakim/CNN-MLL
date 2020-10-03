from PIL import Image
from .layer import Layer
import numpy as np

class Convolutional(Layer) :
  def __init__(self,
               n_filters: int,
               filter_dim: tuple,
               input_shape: tuple = None,
               padding: int = None,
               stride: int = None) :
    """
    Create an instance of convolutional layer

    >>> n_filters
    Number of filters

    >>> filter_dim
    Dimension of filter matrix `(row, col)`

    >>> input_shape
    Input shape `(row, col, channel)`

    >>> padding
    Padding size

    >>> stride
    Stride size
    """
    super().__init__(
      name='conv',
      weights=np.random.uniform(low=-.5, high=.5, size=(n_filters, filter_dim[0], filter_dim[1], input_shape[2])).astype("float") if input_shape is not None else None,
      biases=np.zeros(n_filters),
    )
    self.input = None
    self.n_filters = n_filters
    self.filter_dim = filter_dim
    self.input_shape = input_shape if input_shape is not None else None
    self.input_depth = input_shape[2] if input_shape is not None and len(input_shape) == 3 else None
    self.padding = padding if padding is not None else 0
    self.stride = stride if stride is not None else 1
    self.feature_maps = np.zeros((self.n_filters,
                                  int(((input_shape[0] - filter_dim[0] + 2 * self.padding) / self.stride) + 1),
                                  int(((input_shape[1] - filter_dim[1] + 2 * self.padding) / self.stride) + 1))) if input_shape is not None else None

  def init_weights(self, input_shape) :
    if self.input_depth :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(self.n_filters, self.filter_dim[0], self.filter_dim[1], self.input_depth)).astype("float")
    else :
      self.input_depth = input_shape[0]
      self.weights = np.random.uniform(low=-.5, high=.5, size=(self.n_filters, self.filter_dim[0], self.filter_dim[1], input_shape[0])).astype("float")
      self.feature_maps = np.zeros((self.n_filters,
                                    int(((input_shape[1] - self.filter_dim[0] + 2 * self.padding) / self.stride) + 1),
                                    int(((input_shape[2] - self.filter_dim[1] + 2 * self.padding) / self.stride) + 1)))

  def set_weights(self, weights: np.array) :
    """
    Set convolutional layer weights
    """
    self.weights = weights

  def set_biases(self, biases: np.array) :
    """
    Set convolutional layer biases
    """
    self.biases = biases

  def output_shape(self,
                   input_shape: tuple = None) :
    if self.feature_maps is None :
      self.feature_maps = np.zeros((self.n_filters,
                                    int(((input_shape[1] - self.filter_dim[0] + 2 * self.padding) / self.stride) + 1),
                                    int(((input_shape[2] - self.filter_dim[1] + 2 * self.padding) / self.stride) + 1)))
    return self.feature_maps.shape

  def resize(self, input) :
    """
    If attribute input is an instance of PIL.Image, then use method from PIL library to resize the image
    """
    if Image.isImageType(input) :
      if self.input_shape :
        input.thumbnail((self.input_shape[1], self.input_shape[0]))
        input = np.array(input).transpose(2, 0, 1)
    else :
      pass
    return input

  def crop(self, input) :
    """
    If self has an image attribute, then crop using the method from PIL library, else crop input matrix manually
    """
    if Image.isImageType(input) :
      if self.input_shape :
        image_n_cols, image_n_rows = input.size
        left = (image_n_cols - self.input_shape[1]) // 2
        top = (image_n_rows - self.input_shape[0]) // 2
        right = (image_n_cols + self.input_shape[1]) // 2
        bottom = (image_n_rows + self.input_shape[0]) // 2
        input = input.crop((left, top, right, bottom))
        input = np.array(input).transpose(2, 0, 1)
    else :
      if self.input_shape :
        _, image_n_rows, image_n_cols = input.shape
        input_left = (image_n_cols - self.input_shape[1]) // 2
        input_top = (image_n_rows - self.input_shape[0]) // 2
        input_right = (image_n_cols + self.input_shape[1]) // 2
        input_bottom = (image_n_rows + self.input_shape[0]) // 2

        rows = max((input_bottom - input_top), image_n_rows)
        cols = max((input_right - input_left), image_n_cols)
        res = np.zeros_like(input, shape=(self.input_depth, rows, cols))
        res_left, res_top, res_right, res_bottom = input_left, input_top, input_right, input_bottom

        if rows == image_n_rows :
          res_top = 0
          res_bottom = image_n_rows
        elif rows == input_bottom - input_top :
          res_top = -res_top
          res_bottom = res_top + image_n_rows
          input_top = 0
          input_bottom = rows

        if cols == image_n_cols :
          res_left = 0
          res_right = image_n_cols
        elif cols == input_right - input_left :
          res_left = -res_left
          res_right = res_left + image_n_cols
          input_left = 0
          input_right = cols

        res[:, res_top:res_bottom, res_left:res_right] = input
        input = res[:, input_top:input_bottom, input_left:input_right]
    return input

  def pad(self, input) :
    """
    Add zero-value padding around the input matrix based on the padding size
    """
    left_input = top_input = self.padding
    right_input = bottom_input = padded_input_rows = padded_input_cols = depth = None
    if Image.isImageType(input) :
      right_input, bottom_input = input.size[0] + self.padding, input.size[1] + self.padding
      padded_input_rows, padded_input_cols = input.size[1] + 2 * self.padding, input.size[0] + 2 * self.padding
      depth = len(input.getbands())
      input = Image.fromarray(input)
    else :
      if self.input_shape :
        right_input, bottom_input = self.input_shape[1] + self.padding, self.input_shape[0] + self.padding
        padded_input_rows, padded_input_cols = self.input_shape[0] + 2 * self.padding, self.input_shape[1] + 2 * self.padding
        depth = self.input_depth
      else :
        right_input, bottom_input = input.shape[2] + self.padding, input.shape[1] + self.padding
        padded_input_rows, padded_input_cols = input.shape[1] + 2 * self.padding, input.shape[2] + 2 * self.padding
        depth = input.shape[0]

    result = np.zeros_like(input, shape=(depth, padded_input_rows, padded_input_cols))
    result[:, top_input:bottom_input, left_input:right_input] = input[:, :, :]
    input = result
    return input

  def preprocess(self, input) :
    """
    Resizing, cropping, and add padding to the image based on the input size
    """
    self.input = self.pad(self.crop(self.resize(input)))

  def convolution(self) :
    """
    Convolute input matrix with filters/kernels
    """
    for feature_map_row in range(self.feature_maps.shape[1]) :
      for feature_map_col in range(self.feature_maps.shape[2]) :
        for feature_map_depth in range(self.feature_maps.shape[0]) :
          result = 0
          for filter_row in range(self.weights.shape[1]) :
            for filter_col in range(self.weights.shape[2]) :
              for filter_depth in range(self.weights.shape[3]) :
                result += self.input[filter_depth][feature_map_row * self.stride + filter_row][feature_map_col * self.stride + filter_col] * self.weights[feature_map_depth][filter_row][filter_col][filter_depth]
          result += self.biases[feature_map_depth]
          self.feature_maps[feature_map_depth][feature_map_row][feature_map_col] = result

  def detector(self) :
    """
    Apply ReLU activation function to the feature maps
    """
    for feature_map_row in range(self.feature_maps.shape[1]) :
      for feature_map_col in range(self.feature_maps.shape[2]) :
        for feature_map_depth in range(self.feature_maps.shape[0]) :
          if self.feature_maps[feature_map_depth][feature_map_row][feature_map_col] < 0 :
            self.feature_maps[feature_map_depth][feature_map_row][feature_map_col] = 0

  def forward(self, input):
    """
    Convolution layer forward propagation
    """
    self.preprocess(input)
    self.convolution()
    self.detector()
    return self.feature_maps