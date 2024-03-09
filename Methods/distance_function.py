#One version

from geopy.distance import geodesic

def distance_calculator(coords1, coords2):
    """
    Calculate the geodesic distance between two points given their coordinates.

    Parameters:
    - coords1, coords2 (tuple): Latitude and longitude coordinates of two points.

    Returns:
    - distance (float): The geodesic distance between the two points in kilometers.
    """
    # Create Point objects from latitude and longitude
    point1 = (coords1[0], coords1[1])
    point2 = (coords2[0], coords2[1])

    # Calculate geodesic distance between the two points
    distance = geodesic(point1, point2).kilometers

    return distance
