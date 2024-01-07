def time_to_destination(time_per_unit_distance, distance, start_point, end_point, seasonal_transition):
    """
    Calculates the total time it takes to travel from the start point to the end destination.

    Args:
        time_per_unit_distance (float): The time it takes to travel one unit of distance.
        distance (float): The distance to be traveled.
        start_point (str): The starting point.
        end_point (str): The ending point.
        seasonal_transition (float): The amount of time it takes to transition between the start point and the end point during a seasonal change.

    Returns:
        float: The total time it takes to travel the distance.
    """

    total_time = time_per_unit_distance * distance
    if start_point != end_point:
        total_time += seasonal_transition
    return total_time

def main():
    """
    Prints the total time it takes to travel from one point to another.
    """

    time_per_unit_distance = 1.0  # hours per kilometer
    distance = 100.0  # kilometers
    start_point = "A"
    end_point = "B"
    seasonal_transition = 1.0  # hours

    total_time = time_to_destination(time_per_unit_distance, distance, start_point, end_point, seasonal_transition)
    print(f"The total time it takes to travel {distance} kilometers from {start_point} to {end_point} is {total_time} hours.")

if __name__ == "__main__":
    main()
