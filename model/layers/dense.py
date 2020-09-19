import numpy as np
from .layer import Layer

class Dense(Layer) :
  """
  Flatten stage operation for 2D spatial data

  >>> n_node
  Number of node of the layer

  >>> activation
  Activation function: 'relu' or 'sigmoid', default value is 'sigmoid'
  """
  def __init__(self,
               n_node: int,
               activation: str = 'sigmoid',
               input_shape: int = None) :
    super().__init__(
      name='dense',
      weights= np.zeros(0),
      biases=np.zeros(1)
    )
    self.input_shape = input_shape if not None else None
    self.activation = activation if activation is not None else 'sigmoid'
    self.n_node = n_node
    self.output = np.zeros(n_node)

  def init_weights(self, input):
    if self.input_shape :
      self.weights = np.random.uniform(low=0, high=1, size=(int(self.n_node), self.input_shape)).astype("float")
    else :
      self.weights = np.random.uniform(low=0, high=1, size=(int(self.n_node), len(input))).astype("float")

  def output_shape(self,
                   input_shape: tuple = None) :
    return self.output.shape

  def relu(self, x: float):
    return 0 if x < 0 else x

  def sigmoid(self, x: float):
    return (1 / (1 + np.exp(-x)))

  def backward(self):
    pass

  def forward(self, input) :
    """
    Dense layer forward propagation
    """
    for i in range(self.weights.shape[0]):
      sum = 0
      for j in range(self.weights.shape[1]):
        sum = sum + (input[j] * self.weights[i][j])
      sum = sum + self.biases[0]
      if (self.activation == "relu"):
        self.output[i] = self.relu(sum)
      elif (self.activation == "sigmoid"):
        self.output[i] = self.sigmoid(sum)
    return self.output