import json
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
    """
    Add layer to model
    """
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
    """
    Forward propagation
    """
    output = input
    for layer in self.layers:
      output = layer.forward(output)
    return output

  def predict(self,
              X: np.array,
              y: np.array = None,
              verbose: bool = False) -> list :
    """
    Predict class from an array of inputs.
    Returns an array of predicted classes
    """
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

  def save_model_as_json(self, filename: str) :
    def default(object) :
      """
      Convert Numpy array to list
      """
      if isinstance(object, np.ndarray) :
        return object.tolist()

    model = {'layers': []}
    for layer in self.layers :
      layer_data = {}
      layer_data['name'] = layer.name

      if layer_data['name'] == 'conv' :
        layer_data['n_filters'] = layer.n_filters
        layer_data['filter_dim'] = layer.filter_dim
        layer_data['input_shape'] = layer.input_shape
        layer_data['padding'] = layer.padding
        layer_data['stride'] = layer.stride
        layer_data['weights'] = layer.weights
        layer_data['biases'] = layer.biases

      elif layer_data['name'] == 'dense' :
        layer_data['n_node'] = layer.n_node
        layer_data['activation'] = layer.activation
        layer_data['weights'] = layer.weights
        layer_data['biases'] = layer.biases

      elif layer_data['name'] == 'pool' :
        layer_data['filter_dim'] = layer.filter_dim
        layer_data['stride'] = layer.stride
        layer_data['mode'] = layer.mode

      model['layers'].append(layer_data)

    with open(filename, 'w') as output_file:
      json.dump(model, output_file, indent=4, default=default)

  # def load_model_from_json(self) :
  #   pass