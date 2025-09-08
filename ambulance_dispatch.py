#ambulance dispatch system in python 
# 
# For the ambulance problem, the city can be modeled as a graph, where intersections are nodes and roads are edges with weights representing travel time. The A* algorithm is ideal here because it's both optimal (finds the fastest route) and efficient.

# We can use  Python's heapq module to implement the priority queue, which is essential for A*.

# Problem Simplification:
# The city is a static graph with fixed travel times (costs) on roads.

# We need a heuristic h(n). We'll assign coordinates to each location and use the straight-line distance as our heuristic.
# 
import heapq

def a_star_search(graph, start, goal, coordinates):
    """
    Finds the fastest route using the A* algorithm.
    
    Args explained;
        graph (dict): A dictionary representing the city map. 
                      e.g., {'A': {'B': 5, 'C': 10}} means time from A to B is 5.
        start (str): The starting location (e.g., 'Ambulance_Base').
        goal (str): The destination (e.g., 'Emergency_Location').
        coordinates (dict): A dict with xy-coordinates for each location for the heuristic.

    Returns:
        list of str: The fastest route, or None if no route exists.
    """
    
    def heuristic(node, goal_node):
        # Heuristic function: Straight-line distance (Euclidean distance)
        x1, y1 = coordinates[node]
        x2, y2 = coordinates[goal_node]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # Priority queue stores tuples of (f_score, g_score, current_node, path)
    # We store g_score in the tuple to break ties in f_score, preferring shorter actual paths.
    priority_queue = [(0 + heuristic(start, goal), 0, start, [start])]
    visited = set()

    while priority_queue:
        # Get the node with the lowest f_score
        f_score, g_score, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        
        visited.add(current_node)

        # If we've reached the goal, reconstruct and return the path
        if current_node == goal:
            return path

        # Explore neighbors
        for neighbor, travel_time in graph.get(current_node, {}).items():
            if neighbor not in visited:
                # g(n) is the actual travel time from the start
                new_g_score = g_score + travel_time
                # f(n) = g(n) + h(n)
                new_f_score = new_g_score + heuristic(neighbor, goal)
                heapq.heappush(priority_queue, (new_f_score, new_g_score, neighbor, path + [neighbor]))

    return None

# --- Example Usage ---
# Define the city map as a graph of (Location: {Neighbor: Travel_Time})
city_map = {
    'Ambulance_Base': {'Hospital': 10, 'Intersection_A': 5},
    'Hospital': {'Intersection_B': 3},
    'Intersection_A': {'Hospital': 2, 'Intersection_B': 8, 'Emergency_Location': 20},
    'Intersection_B': {'Emergency_Location': 4},
    'Emergency_Location': {}
}

# Define coordinates for heuristic calculation
location_coords = {
    'Ambulance_Base': (0, 0),
    'Hospital': (4, 3),
    'Intersection_A': (2, 8),
    'Intersection_B': (7, 10),
    'Emergency_Location': (10, 10)
}

start_location = 'Ambulance_Base'
emergency_location = 'Emergency_Location'

fastest_route = a_star_search(city_map, start_location, emergency_location, location_coords)

if fastest_route:
    print(f"Fastest route found from {start_location} to {emergency_location}:")
    print(" -> ".join(fastest_route))
else:
    print(f"No route found from {start_location} to {emergency_location}.")