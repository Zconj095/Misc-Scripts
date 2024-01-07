import math

def calculate_wavelength(tempo, frequency, rhythm):
  """Calculates the "wavelength" of a piece of music given the tempo, frequency, and rhythm.

  Args:
    tempo: The beats per minute.
    frequency: The frequency of the sound wave.
    rhythm: The number of beats per measure.

  Returns:
    The "wavelength" of the music.
  """

  wavelength = 60 / tempo / frequency * rhythm
  return wavelength

if __name__ == "__main__":
  tempo = 120
  frequency = 440
  rhythm = 4
  wavelength = calculate_wavelength(tempo, frequency, rhythm)
  print(f"The 'wavelength' of a piece of music with a tempo of {tempo}, a frequency of {frequency}, and a rhythm of {rhythm} is {wavelength}.")
