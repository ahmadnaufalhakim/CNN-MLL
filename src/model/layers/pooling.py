import numpy as np
from .layer import Layer

class Pooling(Layer):
  """
  Pooling stage operation for 2D spatial data

  >>> filter_dim
  Dimension of filter matrix (row x col)

  >>> stride
  Stride size

  >>> mode
  Pooling mode: 'max' or 'avg', default value is 'max'
  """
  def __init__(self,
               filter_dim: tuple = None,
               stride: int = None,
               mode: str = None) :
    super().__init__(
      name='pool',
      weights=None,
      biases=None
    )
    self.input = None
    self.input_shape = None
    self.filter_dim = filter_dim if filter_dim is not None else 2
    self.stride = stride if stride is not None else 2
    self.mode = mode if mode is not None else "max"
    self.feature_maps = None

  def output_shape(self,
                   input_shape: tuple = None) :
    self.input_shape = input_shape
    self.feature_maps = np.zeros((input_shape[0],
                                  int(((input_shape[1] - self.filter_dim[0]) / self.stride) + 1),
                                  int(((input_shape[2] - self.filter_dim[1]) / self.stride) + 1)))
    return self.feature_maps.shape

  def backward(self, error) :
    """
    Pooling layer backward propagation
    """
    output = np.zeros(self.input_shape)
    if self.mode == "max" :
      for depth in range(error.shape[0]) :
        for row in range(error.shape[1]) :
          for col in range(error.shape[2]) :
            top_row = row * self.stride
            left_col = col * self.stride
            bottom_row, right_col = top_row + self.filter_dim[0], left_col + self.filter_dim[1]
            max_positions = np.where(self.input[:, top_row:bottom_row, left_col:right_col] == self.feature_maps[depth][row][col])
            for max_pos in range(len(max_positions[0])) :
              output[max_positions[0][max_pos]][top_row + max_positions[1][max_pos]][left_col + max_positions[2][max_pos]] = error[depth][row][col]
    elif self.mode == "avg" :
      avg_coefficient = error.shape[1] * error.shape[2]
      for depth in range(error.shape[0]) :
        for row in range(error.shape[1]) :
          for col in range(error.shape[2]) :
            avg_error = error[depth][row][col] / avg_coefficient
            top_row = row * self.stride
            left_col = col * self.stride
            bottom_row, right_col = top_row + self.filter_dim[0], left_col + self.filter_dim[1]
            output[depth, top_row:bottom_row, left_col:right_col] += avg_error
    else :
      pass
    return output

  def forward(self, input) :
    """
    Pooling layer forward propagation
    Reduce spatial size of output from convolution stage to handle overfitting, based on the pooling mode
    """
    self.input = input
    if self.mode == "max" :
      for feature_map_row in range(self.feature_maps.shape[1]) :
        for feature_map_col in range(self.feature_maps.shape[2]) :
          for feature_map_depth in range(self.feature_maps.shape[0]) :
            top_row = feature_map_row * self.stride
            left_col = feature_map_col * self.stride
            bottom_row, right_col = top_row + self.filter_dim[0], left_col + self.filter_dim[1]
            self.feature_maps[feature_map_depth][feature_map_row][feature_map_col] = np.max(self.input[feature_map_depth, top_row:bottom_row, left_col:right_col])
    elif self.mode == "avg" :
      for feature_map_row in range(self.feature_maps.shape[1]) :
        for feature_map_col in range(self.feature_maps.shape[2]) :
          for feature_map_depth in range(self.feature_maps.shape[0]) :
            top_row = feature_map_row * self.stride
            left_col = feature_map_col * self.stride
            bottom_row, right_col = top_row + self.filter_dim[0], left_col + self.filter_dim[1]
            self.feature_maps[feature_map_depth][feature_map_row][feature_map_col] = np.mean(self.input[feature_map_depth, top_row:bottom_row, left_col:right_col])
    else :
      pass
    return self.feature_maps