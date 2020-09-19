import numpy as np
from layer import Layer

class Dense(Layer) :
  """
  Flatten stage operation for 2D spatial data

  >>> input
  An instance of Numpy Array

  >>> n_node
  Number of node of the layer

  >>> activation
  Activation function
  
  """
  def __init__(self, n_node: int, activation: str) :
    super().__init__(
      name='dense',
      weights= np.zeros(0),
      biases=np.zeros(n_node)
    )
    self.activation = activation
    self.n_node = n_node
    self.output = np.zeros(n_node)
    # self.output_shape = 0

  def forward(self, input) :
    print("forward propagation")
    for i in range(self.weights.shape[0]):
      sum = 0
      for j in range(self.weights.shape[1]):
        sum = sum + (input[i] * self.weights[i][j])
      sum = sum + self.biases[i]
      if (self.activation == "relu"):
        self.output[i] = self.relu(sum)
      elif (self.activation == "sigmoid"):
        self.output[i] = self.sigmoid(sum)
    return self.output

  def init_weights(self, input):
    self.weights = np.random.uniform(low=0, high=0.05, size=(int(self.n_node), len(input))).astype("float")
    # self.output_shape = self.weights.shape

  def output_shape(self) :
    print('dense output_shape:', self.output.shape)
    return self.output.shape
       
  def relu(self, x: float):
    return 0 if x < 0 else x

  def sigmoid(self, x: float):
    return (1 / (1 + np.exp(-x)))

  def backward(self):
    print("backward propagation")