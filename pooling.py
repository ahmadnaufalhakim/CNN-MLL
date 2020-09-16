import numpy as np

class Pooling:
  """
  Pooling stage operation for 2D spatial data

  >>> input
  An instance of Numpy Array

  >>> filter_dim
  Dimension of filter matrix (n x n)

  >>> stride
  Stride size

  >>> mode
  Pooling mode: 'max' or 'avg', default value is 'max'
  """
  def __init__(self,
               input,
               filter_dim: int = None,
               stride: int = None,
               mode: str = None) :
    self.input = input
    self.filter_dim = filter_dim if filter_dim is not None else 2
    self.stride = stride if stride is not None else 2
    self.mode = mode if mode is not None else "max"
    self.feature_maps = np.zeros((int(((self.input.shape[0] - self.filter_dim) / self.stride) + 1),
                                  int(((self.input.shape[1] - self.filter_dim) / self.stride) + 1),
                                  self.input.shape[2]),
                                  self.input.dtype)

  def pooling(self) :
    """
    Reduce spatial size of output from convolution stage to handle overfitting, based on the pooling mode
    """
    if self.mode == "max" :
      for feature_map_row in range(self.feature_maps.shape[0]) :
        for feature_map_col in range(self.feature_maps.shape[1]) :
          for feature_map_depth in range(self.feature_maps.shape[2]) :
            top_row = feature_map_row * self.stride
            left_col = feature_map_col * self.stride
            bottom_row, right_col = top_row + self.filter_dim, left_col + self.filter_dim

            self.feature_maps[feature_map_row][feature_map_col][feature_map_depth] = np.max(self.input[top_row:bottom_row, left_col:right_col, feature_map_depth])
    elif self.mode == "avg" :
      for feature_map_row in range(self.feature_maps.shape[0]) :
        for feature_map_col in range(self.feature_maps.shape[1]) :
          for feature_map_depth in range(self.feature_maps.shape[2]) :
            top_row = feature_map_row * self.stride
            left_col = feature_map_col * self.stride
            bottom_row, right_col = top_row + self.filter_dim, left_col + self.filter_dim

            self.feature_maps[feature_map_row][feature_map_col][feature_map_depth] = np.mean(self.input[top_row:bottom_row, left_col:right_col, feature_map_depth])
    else :
      pass