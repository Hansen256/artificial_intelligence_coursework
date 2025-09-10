# Artificial Intelligence Search Algorithms Implementation

This repository contains implementations of different search algorithms as part of an Artificial Intelligence coursework assignment. The work is on how AI systems perform search operations in different scenarios.

## Problem Definitions and Solutions

### 1. Maze Search Problem

#### Problem Definition

The maze problem can be formally defined for a search algorithm as follows:

- **Start State**: The specific cell where the agent begins (e.g., coordinate (0,0) in a 5x5 grid)
- **Goal State**: The target cell the agent needs to reach (e.g., coordinate (4,4))
- **Possible Actions**: From any given cell, the agent can move to an adjacent, non-wall cell
  - Valid moves are {Up, Down, Left, Right}
  - Cannot move into walls or outside grid boundaries

#### Implementation (BFS Solution)

Located in `maze_search.py`, this implementation:

- Uses BFS to find the shortest path through a maze
- Represents the maze as a 2D grid where 0 represents paths and 1 represents walls
- Guarantees the shortest path in terms of steps taken
- Uses Python's `collections.deque` for efficient queue operations

#### How BFS Works

1. **Queue**: Uses a First-In, First-Out data structure to track cells to visit
2. **Initialization**: Adds the start state to the queue
3. **Exploration**: 
   - Removes the first cell from the queue
   - Finds all valid neighbors (reachable via Up, Down, Left, Right)
   - Adds unvisited neighbors to the back of the queue
4. **Termination**: Explores maze level by level until reaching the goal state

### 2. Ambulance Dispatch Problem

#### Solution Analysis

The ambulance dispatch system can be formally defined as follows:

**State Space**:

- Current location of all available ambulances
- Location of the emergency/patient
- Current city traffic conditions
- Status of each ambulance (available, busy, en route)

**Actions**:

- Moving an ambulance between connected nodes on the city's road network
- Each action represents movement to an adjacent intersection

**Goal State**:

- Any state where an ambulance has arrived at the patient's location

**Path Cost**:

- Based on estimated travel time (not just distance)
- Dynamic costs affected by traffic, speed limits, road closures, and time of day

#### Implementation (A* Solution)

Located in `ambulance_dispatch.py`, this implementation:

- Uses A* algorithm to find the optimal route for an ambulance
- Models a city as a graph where:
  - Intersections are nodes
  - Roads are edges with weights (travel time)
  - Uses straight-line (Euclidean) distance as the heuristic function
- Implements a priority queue using Python's `heapq` module
- Finds the fastest route considering both actual travel time and estimated remaining time

#### Why A* is Optimal for This Problem

A* search evaluates paths using the formula: `f(n) = g(n) + h(n)` where:

- `g(n)` is the actual travel time from start to current node
- `h(n)` is the estimated time from current node to goal

**Advantages**:

1. **Optimal & Complete**: Guaranteed to find the fastest route when h(n) doesn't overestimate
2. **Efficient**: Uses heuristic to guide search toward the goal
3. **Adaptable**: Can handle dynamic costs as traffic conditions change

## Implementation Details

### Prerequisites

To run these programs, you need:

- Python 3.x
- Python's standard library (no additional packages required)

### How to Run

#### Maze Solver

1. Open a terminal in the project directory
2. Run the command:

```bash
python maze_search.py
```

This program includes a sample 5x5 maze and will output the path from start (0,0) to goal (4,4) if one exists.

#### Ambulance Dispatch System

1. Open a terminal in the project directory
2. Run the command:

```bash
python ambulance_dispatch.py
```

The program includes a sample city map with various locations (Ambulance Base, Hospital, etc.) and will output the fastest route from the Ambulance Base to the Emergency Location.

### Sample Output

#### Maze Solution

The program will output the coordinates of the path found, for example:

```text
Path found from (0, 0) to (4, 4):
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
```

#### Ambulance Dispatch Route

The program will output the sequence of locations in the fastest route, for example:

```text
Fastest route found from Ambulance_Base to Emergency_Location:
Ambulance_Base -> Intersection_A -> Intersection_B -> Emergency_Location
```

## Project Information

### Repository Structure

```text
artificial_intelligence_coursework/
├── maze_search.py
├── ambulance_dispatch.py
└── README.md
```

### Author

Tumusiime Hansen Andrew 

### Course Context

This work is part of an Artificial Intelligence course unit focusing on how AI systems perform search operations in different problem spaces. The implementations demonstrate practical applications of BFS and A* search algorithms in solving real-world problems.
