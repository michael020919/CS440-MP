from abc import ABC, abstractmethod
from typing import List
from itertools import count
import math
state_counter = count()


class SearchState(ABC):
    """Abstract base class for all search states"""
    def __init__(self, current_state, target_state, path_cost=0, enable_heuristic=True):
        self.current_state = current_state
        self.target_state = target_state
        # Unique identifier for tie-breaking in priority queue
        self.creation_order = next(state_counter)
        # g(n) - actual cost from start to current state
        self.path_cost = path_cost
        self.enable_heuristic = enable_heuristic
        if enable_heuristic:
            self.heuristic_value = self.calculate_heuristic()
        else:
            self.heuristic_value = 0

    # Generate all valid successor states from current state
    @abstractmethod
    def generate_successors(self):
        pass
    
    # Check if current state satisfies goal condition
    @abstractmethod
    def goal_test(self):
        pass
    
    # Calculate heuristic estimate h(n) from current state to goal
    @abstractmethod
    def calculate_heuristic(self):
        pass
    
    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.path_cost + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using creation_order as below
        # so that the state created later is considered "less than" the one created earlier
        if self.creation_order > other.creation_order:
            return True


    # Hash function for visited state tracking
    @abstractmethod
    def __hash__(self):
        pass
    
    # Equality check for state comparison
    @abstractmethod
    def __eq__(self, other_state):
        pass
    

# current_state: a length 3 list indicating the centroid location in the grid and the robot's shape
# goals: a list of tuples of (x,y) centroid coordinates in the grid that have not yet been reached
# maze: a maze object (deals with checking collision with walls...)
class MazeState(SearchState):
    def __init__(self, current_state, goals, path_cost, maze, enable_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally,
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors
        super().__init__(current_state, goals, path_cost, enable_heuristic)

    # @TODO(VI)
    def generate_successors(self):
        # if the shape changes, it will have a const cost of 10.
        # otherwise, the move cost will be the euclidean distance between the start and the end positions
        nbr_states = []
        return nbr_states

    # @TODO(VI)
    def goal_test(self):
        return True

    # We hash BOTH the state and the remaining goals
    #   This is because (x, y, h, (goal A, goal B)) is different from (x, y, h, (goal A))
    #   In the latter we've already visited goal B, changing the nature of the remaining search
    # NOTE: the order of the goals in self.target_state matters, needs to remain consistent
    # @TODO(VI)
    def __hash__(self):
        return 0

    # @TODO(VI)
    def __eq__(self, other):
        return True

    # Our heuristic is: distance(self.state, nearest_goal)
    # We use euclidean distance
    # @TODO(VI)
    def calculate_heuristic(self):
        return 0

    # This method allows the heap to sort States according to f = g + h value
    # @TODO(VI)
    def __lt__(self, other):
        pass

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.target_state)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.target_state)
