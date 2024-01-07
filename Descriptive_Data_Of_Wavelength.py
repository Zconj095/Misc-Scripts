import math

def wavelength_frequency(wavelength, frequency):
    """
    Calculates the wavelength of a sound wave given its frequency.

    Args:
        wavelength (float): The wavelength of the sound wave, in meters.
        frequency (float): The frequency of the sound wave, in hertz.

    Returns:
        float: The wavelength of the sound wave, in meters.
    """

    speed_of_sound = 343.0  # meters per second
    return speed_of_sound / frequency

def pitch_tempo(pitch, tempo):
    """
    Calculates the pitch of a piece of music given its tempo.

    Args:
        pitch (float): The pitch of the piece of music, in hertz.
        tempo (float): The tempo of the piece of music, in beats per minute.

    Returns:
        float: The pitch of the piece of music, in hertz.
    """

    return pitch * tempo

def main():
    """
    Prints the wavelength and pitch of a sound wave.
    """

    wavelength = 1.0  # meters
    frequency = 343.0  # hertz
    print(wavelength_frequency(wavelength, frequency))

    pitch = 440.0  # hertz
    tempo = 120.0  # beats per minute
    print(pitch_tempo(pitch, tempo))

if __name__ == "__main__":
    main()
