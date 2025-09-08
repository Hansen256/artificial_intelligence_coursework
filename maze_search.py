#A program to solve a maze problem using a breadth search algorithm while representing the Grid a a 2D list 
# we shall use the breadth first search in finding the shortest path in terms of steps because it explores layer by layer.
#We'll use Python's collections.deque as an efficient queue.

from collections import deque
def solveMaze(maze,start,goal):
    """
    arguments explanation;
    maze (list of lists): The maze grid where 0 is a path and 1 is a wall.
    start (tuple): The starting coordinates (row, col).
    goal (tuple): The goal coordinates (row, col).
    
    """
    rows, cols = len(maze), len(maze[0]) # this line finds out(and defines ) the boundaries of the maze to make sure our program does not stray outside it  

    # The queue will store tuples of (path)
    queue = deque([[start]])

    # A set to keep track of visited cells to avoid cycles and redundant work
    visited = {start}

    while queue:
        # Get the current path from the front of the queue
        path = queue.popleft()
        current_cell = path[-1]

        # If we've reached the goal, return the path
        if current_cell == goal:
            return path

        # Explore neighbors: Up, Down, Left, Right
        row, col = current_cell
        possible_moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

        for move in possible_moves:
            r, c = move
            # Check if the move is valid (within bounds, not a wall, and not visited)
            if 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0 and move not in visited:
                visited.add(move)
                # Create a new path and add it to the queue
                new_path = list(path)
                new_path.append(move)
                queue.append(new_path)

    # If the queue becomes empty and goal was not found, no path exists
    return None

# --- Example Usage ---
# Define a 5x5 maze: 0 = path, 1 = wall
my_maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start_pos = (0, 0)
goal_pos = (4, 4)

solution_path = solveMaze(my_maze, start_pos, goal_pos)

if solution_path:
    print(f"Path found from {start_pos} to {goal_pos}:")
    print(solution_path)
else:
    print(f"No path found from {start_pos} to {goal_pos}.")


