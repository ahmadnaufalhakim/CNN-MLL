import numpy as np
from layer import Layer

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
    )
    self.input = None
    self.filter_dim = filter_dim if filter_dim is not None else 2
    self.stride = stride if stride is not None else 2
    self.mode = mode if mode is not None else "max"
    self.feature_maps = None

  def output_shape(self,
                   input_shape: tuple = None) :
    self.feature_maps = np.zeros((input_shape[0],
                                  int(((input_shape[1] - self.filter_dim[0]) / self.stride) + 1),
                                  int(((input_shape[2] - self.filter_dim[1]) / self.stride) + 1)))
    print('pooling output_shape:', self.feature_maps.shape)
    return self.feature_maps.shape

  def forward(self, input) :
    """
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