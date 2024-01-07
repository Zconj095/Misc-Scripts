import numpy as np

def displacement(vector_location, length, width, vertice_vector_count):
  """Calculates the displacement of a vector location.

  Args:
    vector_location: The vector location of the object.
    length: The length of the object.
    width: The width of the object.
    vertice_vector_count: The number of vertices in the object.

  Returns:
    The displacement of the object.
  """

  vertice_vector_count_integer = np.int64(vertice_vector_count)
  length_integer = np.int64(length)
  displacement = np.dot(vector_location, length_integer) * math.pow(width, 2) * vertice_vector_count_integer

  return displacement

if __name__ == "__main__":
  vector_location = (0, 0, 0)
  length = 10.5
  width = 5
  vertice_vector_count = 8

  displacement = displacement(vector_location, length, width, vertice_vector_count)
  print(displacement)
