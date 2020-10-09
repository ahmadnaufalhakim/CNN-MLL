import numpy as np
from .layer import Layer

class Flatten(Layer) :
  """
  Flatten stage operation for 2D spatial data
  """
  def __init__(self) :
    super().__init__(
      name='flatten',
      weights=None,
      biases=None
    )
    self.output = np.zeros(0)
    self.input_shape = None

  def output_shape(self, input_shape) :
    self.input_shape = input_shape
    self.output = np.zeros(np.prod(input_shape))
    return self.output.shape

  def backward(self, error, learning_rate = None, momentum = None) :
    """
    Flatten layer backward propagation
    """
    output = np.array(error)
    return output.reshape(self.input_shape)

  def forward(self, input) :
    """
    Flatten layer forward propagation
    """
    self.output = input.flatten()
    return self.output