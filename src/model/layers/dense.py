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

  >>> input_shape
  Input shape
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
    self.input = None
    self.input_shape = input_shape if not None else None
    self.activation = activation if activation is not None else 'sigmoid'
    self.n_node = n_node
    self.output = np.zeros(n_node)
    self.delta_weights = None

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
    self.input_shape = input_shape[0]
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

  def backward(self, error, learning_rate, momentum) :
    """
    Dense layer backward propagation
    """
    err = self.derivative_activation(error)
    if self.delta_weights is not None :
      self.delta_weights = learning_rate * self.derivative_weight(err) + momentum * self.delta_weights
    else :
      self.delta_weights = momentum * self.derivative_weight(err)
    return self.derivative_input(err)

  def derivative_activation(self, error) :
    """
    Compute derivative of activation function (d_E/d_net)
    """
    def derivative_relu(error) :
      """
      Compute derivative of ReLU activation function
      """
      for i in range(len(error)) :
        error[i] = 0 if error[i] < 0 else error[i]
        return error

    def derivative_sigmoid(error) :
      """
      Compute derivative of sigmoid activation function
      """
      for i in range(len(error)) :
        error[i] = error[i] * (1 - error[i])
        return error

    if self.activation == 'relu' :
      return derivative_relu(error)
    elif self.activation == 'sigmoid' :
      return derivative_sigmoid(error)

  def derivative_input(self, error) :
    """
    Compute derivative of dense layer's output with respect to its inputs
    """
    return np.dot(self.weights.transpose(), error)

  def derivative_weight(self, error) :
    """
    Compute derivative of dense layer's output with respect to its weights
    """
    input = np.zeros(self.input_shape + 1)
    input[0] = 1
    input[1:] = self.input
    return np.dot(error, input.reshape(1, self.input_shape + 1))

  def forward(self, input) :
    """
    Dense layer forward propagation
    """
    self.input = input
    temp_output = np.dot(self.weights, self.input)
    # if (np.dot(self.weights, input).shape[0] == 1) :
    #   print(temp_output + self.biases)
    if (self.activation == "relu"):
      for node in range(self.n_node) :
        self.output[node] = self.relu(temp_output[node] + self.biases[node])
    elif (self.activation == "sigmoid"):
      for node in range(self.n_node) :
        self.output[node] = self.sigmoid(temp_output[node] + self.biases[node])
    return self.output

  def update_weights(self) :
    """
    Update dense layer's biases and weights using negative gradient
    """
    self.biases -= self.delta_weights[:, 0]
    self.weights -= self.delta_weights[:, 1:]
    self.delta_weights = None