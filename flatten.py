import numpy as np
from layer import Layer

class Flatten(Layer) :
  """
  Flatten stage operation for 2D spatial data

  >>> input
  An instance of Numpy Array
  """
  def __init__(self) :
    super().__init__(
      name='flatten',
      weights=None,
      biases=None
    )
    self.output = np.zeros(0)

  def forward(self, input) :
    self.output = input.flatten()
    # self.output_shape = len(self.output)
    return self.output

  def output_shape(self) :
    print('flatten output_shape:', self.output.shape)
    return self.output.shape