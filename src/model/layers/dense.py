import numpy as np
from .layer import Layer

np.seterr(all='ignore')

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
      weights=np.zeros(0),
      biases=np.zeros(n_node)
    )
    self.input_shape = input_shape if not None else None
    self.activation = activation if activation is not None else 'sigmoid'
    self.n_node = n_node
    self.output = np.zeros(n_node)

  def init_weights(self, input) :
    if self.input_shape :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), self.input_shape)).astype("float")
    else :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), input[0])).astype("float")

  def set_weights(self, weights: np.array) :
    """
    Set dense layer weights
    """
    self.weights = weights

  def set_biases(self, biases: np.array) :
    """
    Set dense layer biases
    """
    self.biases = biases

  def output_shape(self,
                   input_shape: tuple = None) :
    return self.output.shape

  def relu(self, input) :
    """
    Apply ReLU function
    """
    input = 0 if input < 0 else input
    return input

  def sigmoid(self, input) :
    """
    Apply sigmoid function
    """
    input = 1 / (1 + np.exp(-input))
    return input

  def backward(self):
    pass

  def forward(self, input) :
    """
    Dense layer forward propagation
    """
    temp_output = np.dot(self.weights, input)
    # if (np.dot(self.weights, input).shape[0] == 1) :
    #   print(temp_output + self.biases)
    if (self.activation == "relu"):
      for node in range(self.n_node) :
        self.output[node] = self.relu(temp_output[node] + self.biases[node])
    elif (self.activation == "sigmoid"):
      for node in range(self.n_node) :
        self.output[node] = self.sigmoid(temp_output[node] + self.biases[node])
    return self.output