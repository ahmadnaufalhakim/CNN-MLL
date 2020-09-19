import numpy as np
from .layer import Layer

np.seterr(all='ignore')

def relu(input) :
  for i in range(len(input)) :
    input[i] = 0 if input[i] < 0 else input[i]
  return input

def sigmoid(input) :
  for i in range(len(input)) :
    input[i] = 1 / (1 + np.exp(-input[i]))
  return input

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
      biases=0
    )
    self.input_shape = input_shape if not None else None
    self.activation = activation if activation is not None else 'sigmoid'
    self.n_node = n_node
    self.output = np.zeros(n_node)

  def init_weights(self, input):
    if self.input_shape :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), self.input_shape)).astype("float")
    else :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), input[0])).astype("float")

  def output_shape(self,
                   input_shape: tuple = None) :
    return self.output.shape

  def backward(self):
    pass

  def forward(self, input) :
    """
    Dense layer forward propagation
    """
    temp_output = np.dot(self.weights, input)
    if (np.dot(self.weights, input).shape[0] == 1) :
      print(temp_output + self.biases)

    if (self.activation == "relu"):
      self.output = relu(temp_output + self.biases)
    elif (self.activation == "sigmoid"):
      self.output = sigmoid(temp_output + self.biases)
    return self.output