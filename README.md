# Artificial Intelligence Search Algorithms Implementation

This repository contains implementations of different search algorithms as part of an Artificial Intelligence coursework assignment. The work is on  how AI systems perform search operations in different scenarios.

## Implemented Algorithms

### 1. Breadth-First Search (BFS) - Maze Solver

Located in `maze_search.py`, this implementation:

- Uses BFS to find the shortest path through a maze
- Represents the maze as a 2D grid where 0 represents paths and 1 represents walls
- Guarantees the shortest path in terms of steps taken
- Uses Python's `collections.deque` for efficient queue operations

### 2. A* Search - Ambulance Dispatch System

Located in `ambulance_dispatch.py`, this implementation:

- Uses A* algorithm to find the optimal route for an ambulance
- Models a city as a graph where:
  - Intersections are nodes
  - Roads are edges with weights (travel time)
  - Uses straight-line (Euclidean) distance as the heuristic function
- Implements a priority queue using Python's `heapq` module
- Finds the fastest route considering both actual travel time and estimated remaining time

## Prerequisites

To run these programs, you need:

- Python 3.x
- Python's standard library (no additional packages required)

## How to Run

### Maze Solver

1. Open a terminal in the project directory
2. Run the command:

```bash
python maze_search.py
```

This  program includes a sample 5x5 maze and will output the path from start (0,0) to goal (4,4) if one exists.

### Ambulance Dispatch System

1. Open a terminal in the project directory
2. Run the command:

```bash
python ambulance_dispatch.py
```

The program includes a sample city map with various locations (Ambulance Base, Hospital, etc.) and will output the fastest route from the Ambulance Base to the Emergency Location.

## Sample Output

### Maze Solution

The program will output the coordinates of the path found, for example:

```text
Path found from (0, 0) to (4, 4):
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
```

### Ambulance Dispatch Route

The program will output the sequence of locations in the fastest route, for example:

```text
Fastest route found from Ambulance_Base to Emergency_Location:
Ambulance_Base -> Intersection_A -> Intersection_B -> Emergency_Location
```

## Repository Structure

```text
artificial_intelligence_coursework/
├── maze_search.py
├── ambulance_dispatch.py
└── README.md
```

## Author

Tumusiime Hansen Andrew 

## Course Context

This work is part of an Artificial Intelligence course unit focusing on how AI systems perform search operations in different problem spaces.
