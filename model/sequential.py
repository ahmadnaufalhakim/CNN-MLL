import numpy as np
import sys
import time

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

  def predict(self,
              X: np.array,
              y: np.array = None,
              verbose: bool = False) -> list :
    predicted_labels = []
    print('Found', len(X), 'data to be predicted', end='\n\n')

    if self.layers[-1].name == 'dense' :
      if self.layers[-1].activation == 'relu' :
        for idx, input_data in enumerate(X) :
          print('Predicting data #' + str(idx + 1), '...', end='\n' if verbose else ' ')
          sys.stdout.flush()
          start = time.time()
          output = self.forward(input_data)
          finish = time.time()
          result = np.where(output == np.max(output))
          label = result[0][0] if len(result[0]) == 1 else result[0][np.random.randint(len(result[0]))]
          predicted_labels.append(label)
          if verbose :
            print('\nModel prediction:', label, end='\t| ')
            if y :
              print('Correct label:', y[idx], end='\t| ')
            print('Raw output:', output)
            print('Model finished in:', finish-start, 'seconds', end='\n\n')
          else :
            print('done')

      elif self.layers[-1].activation == 'sigmoid' :
        for idx, input_data in enumerate(X) :
          print('Predicting data #' + str(idx + 1), '...', end='\n' if verbose else ' ')
          sys.stdout.flush()
          start = time.time()
          output = self.forward(input_data)
          finish = time.time()
          label = 0 if output <= 0.5 else 1
          predicted_labels.append(label)
          if verbose :
            print('Model prediction:', label, end='\t| ')
            if y :
              print('Correct label:', y[idx], end='\t| ')
            print('Raw output:', output)
            print('Model finished in:', finish-start, 'seconds', end='\n\n')
          else :
            print('done')

      else :
        raise Exception('Invalid activation function: ' + self.layers[-1].activation)

    else :
      raise Exception('Invalid layer type: ' + self.layers[-1].name)

    return predicted_labels