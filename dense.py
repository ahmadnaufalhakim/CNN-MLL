import numpy as np

class Dense:
  """
  Flatten stage operation for 2D spatial data

  >>> input
  An instance of Numpy Array

  >>> n_node
  Number of node of the layer

  >>> activation
  Activation function
  
  """
  def __init__(self, input, n_node: int, activation: str):
    self.name = "dense"
    self.activation = activation
    self.input = input
    self.n_node = n_node
    self.weights = np.random.uniform(low=0, high=0.05, size=(int(n_node), len(input))).astype("float")
    self.output = np.zeros(n_node)
    self.biases = np.zeros(n_node)

  def forward(self):
    print("forward propagation")
    for i in range(self.weights.shape[0]):
      sum = 0
      for j in range(self.weights.shape[1]):
        sum = sum + (self.input[i] * self.weights[i][j])
      sum = sum + self.biases[i]
      if (self.activation == "relu"):
        self.output[i] = self.relu(sum)
      elif (self.activation == "sigmoid"):
        self.output[i] = self.sigmoid(sum)
       
  def relu(self, x: float):
    return 0 if x < 0 else x

  def sigmoid(self, x: float):
    return (1 / (1 + np.exp(-x)))

  def backward():
    print("backward propagation")