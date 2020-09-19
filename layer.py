import numpy as np

class Layer :
  def __init__(self,
               name = None,
               weights = None,
               biases = None) :
    self.name = name if name is not None else None
    self.weights = weights if weights is not None else None
    self.biases = biases if biases is not None else None

  def forward(self, input) -> np.array:
    pass

# class Convolution(Layer):
#   def forward(self, input):
#     print("forward")

# class Pooling(Layer):
#   def forward(self, input):
#     print("forward")

# class Flatten(Layer):
#   def forward(self, input):
#     print("forward")
    
# class Dense(Layer):
#   def forward(self, input):
#     print("forward")