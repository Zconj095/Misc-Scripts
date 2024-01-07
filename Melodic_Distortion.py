import math

def calculate_melodic_distortion(bpm, rhythm):
  """Calculates the melodic distortion of a piece of music given the BPM and rhythm.

  Args:
    bpm: The beats per minute.
    rhythm: The range of wavelengths.

  Returns:
    The melodic distortion.
  """

  wavelength = 60 / bpm
  distortion = rhythm / wavelength
  return distortion

if __name__ == "__main__":
  bpm = 120
  rhythm = 4
  distortion = calculate_melodic_distortion(bpm, rhythm)
  print(f"The melodic distortion of a piece of music with a BPM of {bpm} and a rhythm of {rhythm} is {distortion}.")
