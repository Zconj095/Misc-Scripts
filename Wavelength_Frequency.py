import math

def wavelength_frequency(wavelength, frequency):
  speed_of_sound = 343
  return speed_of_sound / frequency

def pitch_tempo(pitch, tempo):
  return pitch * tempo

def main():
  wavelength = 1
  frequency = 343
  print(wavelength_frequency(wavelength, frequency))

  pitch = 440
  tempo = 120
  print(pitch_tempo(pitch, tempo))

if __name__ == "__main__":
  main()
