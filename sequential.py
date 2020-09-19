import numpy as np

class SequentialModel:
  def __init__(self, layers):
    self.layers = []
    self.output_shapes = []
    self.weighted_layers = ['conv', 'dense']

    for layer in layers:
      self.add(layer)

  def add(self, layer):
    # init_weights = getattr(layer, 'init_weights', None)

    if len(self.layers) :
      # if callable(init_weights):
      if layer.name in self.weighted_layers:
        layer.init_weights(self.output_shapes[-1])
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