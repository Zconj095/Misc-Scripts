def frequency_pattern(new_timeframe, old_timeframe, new_old_timeframe, begin_point, start_point, beginning_location, end_location):
  """
  Calculates the frequency pattern.

  Args:
    new_timeframe: The new timeframe.
    old_timeframe: The old timeframe.
    new_old_timeframe: The new old timeframe.
    begin_point: The begin point.
    start_point: The start point.
    beginning_location: The beginning location.
    end_location: The end location.

  Returns:
    The frequency pattern.
  """

  frequency_pattern = new_timeframe * old_timeframe * new_old_timeframe * (begin_point * start_point) * beginning_location + end_location
  return frequency_pattern

if __name__ == "__main__":
  new_timeframe = 10
  old_timeframe = 20
  new_old_timeframe = 30
  begin_point = 40
  start_point = 50
  beginning_location = 60
  end_location = 70

  frequency_pattern = frequency_pattern(new_timeframe, old_timeframe, new_old_timeframe, begin_point, start_point, beginning_location, end_location)
  print(frequency_pattern)
