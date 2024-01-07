def calculate_time(distance, velocity, bpm, rhythm):
  """Calculates the time it takes for an object to travel a certain distance at a given velocity, with a given BPM (beats per minute) and rhythm.

  Args:
    distance: The distance that the object needs to travel.
    velocity: The velocity of the object.
    bpm: The beats per minute.
    rhythm: The number of beats per measure.

  Returns:
    The time it takes for the object to travel the distance.
  """

  time = distance / velocity * bpm / rhythm
  return time

if __name__ == "__main__":
  distance = 100
  velocity = 10
  bpm = 120
  rhythm = 4
  time = calculate_time(distance, velocity, bpm, rhythm)
  print(f"The time it takes to travel {distance} meters at a velocity of {velocity} meters per second, with a BPM of {bpm} and a rhythm of {rhythm} beats per measure is {time} seconds.")
