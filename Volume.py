def volume(unit_size, number_of_faces, number_of_vertices, mass_amount):
  """Calculates the volume of a 3D object.

  Args:
    unit_size: The size of a single unit in the object's coordinate system.
    number_of_faces: The number of faces in the object.
    number_of_vertices: The number of vertices in the object.
    mass_amount: The mass of the object.

  Returns:
    The volume of the object.
  """

  return unit_size * number_of_faces * number_of_vertices * mass_amount

if __name__ == "__main__":
  unit_size = 1
  number_of_faces = 6
  number_of_vertices = 8
  mass_amount = 1

  volume = volume(unit_size, number_of_faces, number_of_vertices, mass_amount)
  # Print a more descriptive output.
  print(f"The volume of the object is {volume} cubic {unit_size}.")
  print(volume)
