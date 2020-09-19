import numpy as np

class SequentialModel:
  """
  Create a basic model instance

  >>> layers
  An array of layers
  """
  def __init__(self, layers: list):
    self.layers = []
    self.output_shapes = []
    self.weighted_layers = ['conv', 'dense']

    for layer in layers:
      self.add(layer)

  def add(self, layer):
    if len(self.layers) :
      if layer.name in self.weighted_layers:
        layer.init_weights(self.output_shapes[-1])
      self.output_shapes.append(layer.output_shape(self.output_shapes[-1]))
    else :
      if layer.name in self.weighted_layers:
        layer.init_weights(layer.input_shape)
      self.output_shapes.append(layer.output_shape())

    self.layers.append(layer)

  def forward(self, input: np.array):
    output = input
    for layer in self.layers:
      output = layer.forward(output)
    return output