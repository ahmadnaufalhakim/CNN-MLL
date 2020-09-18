class Flatten:
  """
  Flatten stage operation for 2D spatial data

  >>> input
  An instance of Numpy Array
  """
  def __init__(self, input):
    self.name = "flatten"
    self.input = input
    self.output = input.flatten()