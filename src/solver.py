import time
import random
import logging
from src.maze import Maze
import heapq
import logging

logging.basicConfig(level=logging.DEBUG)


class Solver(object):
    """Base class for solution methods.
    Every new solution method should override the solve method.

    Attributes:
        maze (list): The maze which is being solved.
        neighbor_method:
        quiet_mode: When enabled, information is not outputted to the console

    """

    def __init__(self, maze, quiet_mode, neighbor_method):
        logging.debug("Class Solver ctor called")

        self.maze = maze
        self.neighbor_method = neighbor_method
        self.name = ""
        self.quiet_mode = quiet_mode

    def solve(self):
        logging.debug('Class: Solver solve called')
        raise NotImplementedError

    def get_name(self):
        logging.debug('Class Solver get_name called')
        raise self.name

    def get_path(self):
        logging.debug('Class Solver get_path called')
        return self.path


class BreadthFirst(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class BreadthFirst ctor called')

        self.name = "Breadth First Recursive"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):

        """Function that implements the breadth-first algorithm for solving the maze. This means that
                for each iteration in the outer loop, the search visits one cell in all possible branches. Then
                moves on to the next level of cells in each branch to continue the search."""

        logging.debug("Class BreadthFirst solve called")
        current_level = [self.maze.entry_coor]  # Stack of cells at current level of search
        path = list()  # To track path of solution cell coordinates

        print("\nSolving the maze with breadth-first search...")
        time_start = time.clock()

        while True:  # Loop until return statement is encountered
            next_level = list()

            while current_level:  # While still cells left to search on current level
                k_curr, l_curr = current_level.pop(0)  # Search one cell on the current level
                self.maze.grid[k_curr][l_curr].visited = True  # Mark current cell as visited
                path.append(((k_curr, l_curr), False))  # Append current cell to total search path

                if (k_curr, l_curr) == self.maze.exit_coor:  # Exit if current cell is exit cell
                    if not self.quiet_mode:
                        print("Number of moves performed: {}".format(len(path)))
                        print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                    print(path)
                    return path

                neighbour_coors = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indicies
                neighbour_coors = self.maze.validate_neighbours_solve(neighbour_coors, k_curr,
                                                                  l_curr, self.maze.exit_coor[0],
                                                                  self.maze.exit_coor[1], self.neighbor_method)

                if neighbour_coors is not None:
                    for coor in neighbour_coors:
                        next_level.append(coor)  # Add all existing real neighbours to next search level

            for cell in next_level:
                current_level.append(cell)  # Update current_level list with cells for nex search level
        logging.debug("Class BreadthFirst leaving solve")


class BiDirectional(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class BiDirectional ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Bi Directional"

    def solve(self):

        """Function that implements a bidirectional depth-first recursive backtracker algorithm for
        solving the maze, i.e. starting at the entry point and exit points where each search searches
        for the other search path. NOTE: THE FUNCTION ENDS IN AN INFINITE LOOP FOR SOME RARE CASES OF
        THE INPUT MAZE. WILL BE FIXED IN FUTURE."""
        logging.debug("Class BiDirectional solve called")

        grid = self.maze.grid
        k_curr, l_curr = self.maze.entry_coor            # Where to start the first search
        p_curr, q_curr = self.maze.exit_coor             # Where to start the second search
        grid[k_curr][l_curr].visited = True    # Set initial cell to visited
        grid[p_curr][q_curr].visited = True    # Set final cell to visited
        backtrack_kl = list()                  # Stack of visited cells for backtracking
        backtrack_pq = list()                  # Stack of visited cells for backtracking
        path_kl = list()                       # To track path of solution and backtracking cells
        path_pq = list()                       # To track path of solution and backtracking cells

        if not self.quiet_mode:
            print("\nSolving the maze with bidirectional depth-first search...")
        time_start = time.clock()

        while True:   # Loop until return statement is encountered
            neighbours_kl = self.maze.find_neighbours(k_curr, l_curr)    # Find neighbours for first search
            real_neighbours_kl = [neigh for neigh in neighbours_kl if not grid[k_curr][l_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_kl = [neigh for neigh in real_neighbours_kl if not grid[neigh[0]][neigh[1]].visited]

            neighbours_pq = self.maze.find_neighbours(p_curr, q_curr)    # Find neighbours for second search
            real_neighbours_pq = [neigh for neigh in neighbours_pq if not grid[p_curr][q_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_pq = [neigh for neigh in real_neighbours_pq if not grid[neigh[0]][neigh[1]].visited]

            if len(neighbours_kl) > 0:   # If there are unvisited neighbour cells
                backtrack_kl.append((k_curr, l_curr))              # Add current cell to stack
                path_kl.append(((k_curr, l_curr), False))          # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbours_kl)      # Choose random neighbour
                grid[k_next][l_next].visited = True                # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(backtrack_kl) > 0:                  # If there are no unvisited neighbour cells
                path_kl.append(((k_curr, l_curr), True))   # Add coordinates to part of search path
                k_curr, l_curr = backtrack_kl.pop()        # Pop previous visited cell (backtracking)

            if len(neighbours_pq) > 0:                        # If there are unvisited neighbour cells
                backtrack_pq.append((p_curr, q_curr))           # Add current cell to stack
                path_pq.append(((p_curr, q_curr), False))       # Add coordinates to part of search path
                p_next, q_next = random.choice(neighbours_pq)   # Choose random neighbour
                grid[p_next][q_next].visited = True             # Move to that neighbour
                p_curr = p_next
                q_curr = q_next

            elif len(backtrack_pq) > 0:                  # If there are no unvisited neighbour cells
                path_pq.append(((p_curr, q_curr), True))   # Add coordinates to part of search path
                p_curr, q_curr = backtrack_pq.pop()        # Pop previous visited cell (backtracking)

            # Exit loop and return path if any opf the kl neighbours are in path_pq.
            if any((True for n_kl in real_neighbours_kl if (n_kl, False) in path_pq)):
                path_kl.append(((k_curr, l_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  # Zip paths
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                logging.debug("Class BiDirectional leaving solve")
                return path

            # Exit loop and return path if any opf the pq neighbours are in path_kl.
            elif any((True for n_pq in real_neighbours_pq if (n_pq, False) in path_kl)):
                path_pq.append(((p_curr, q_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  # Zip paths
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                logging.debug("Class BiDirectional leaving solve")
                return path


class DepthFirstBacktracker(Solver):
    """A solver that implements the depth-first recursive backtracker algorithm.
    """

    def __init__(self, maze, quiet_mode=False,  neighbor_method="fancy"):
        logging.debug('Class DepthFirstBacktracker ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Depth First Backtracker"

    def solve(self):
        logging.debug("Class DepthFirstBacktracker solve called")
        k_curr, l_curr = self.maze.entry_coor      # Where to start searching
        self.maze.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visited_cells = list()                  # Stack of visited cells for backtracking
        path = list()                           # To track path of solution and backtracking cells
        if not self.quiet_mode:
            print("\nSolving the maze with depth-first search...")

        time_start = time.time()

        while (k_curr, l_curr) != self.maze.exit_coor:     # While the exit cell has not been encountered
            neighbour_indices = self.maze.find_neighbours(k_curr, l_curr)    # Find neighbour indices
            neighbour_indices = self.maze.validate_neighbours_solve(neighbour_indices, k_curr,
                l_curr, self.maze.exit_coor[0], self.maze.exit_coor[1], self.neighbor_method)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                path.append(((k_curr, l_curr), False))  # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbour_indices)   # Choose random neighbour
                self.maze.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(visited_cells) > 0:              # If there are no unvisited neighbour cells
                path.append(((k_curr, l_curr), True))   # Add coordinates to part of search path
                k_curr, l_curr = visited_cells.pop()    # Pop previous visited cell (backtracking)

        path.append(((k_curr, l_curr), False))  # Append final location to path
        if not self.quiet_mode:
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        logging.debug('Class DepthFirstBacktracker leaving solve')
        print(path)
        return path

class UniformCostSearch(Solver):      #UCS Class
    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class UniformCostSearch ctor called')
        self.name = "uniformcostsearch"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):
        logging.debug("Class UniformCostSearch solve called")
        start = self.maze.entry_coor
        goal = self.maze.exit_coor
        frontier = []   #다음 방문해야 하는 목록 중 아직 방문하지 않은 목록
        heapq.heappush(frontier, (0, start, [start]))  # (cost, position, path)
        visited = set() #방문한 목록들의 집합.
        all_visited_cells = []  #모든 방문한 셀을 저장하는 리스트

        while frontier:
            current_cost, current, path = heapq.heappop(frontier)

            if current in visited:
                tt=(current, False)
                all_visited_cells.append(tt)
                continue
            visited.add(current)
            ff=(current, True)
            all_visited_cells.append(ff)    #output

            if current == goal:
                print(f"Cost: {current_cost}, Path: {path}")
                print("Searched Cells: ", all_visited_cells)
                return all_visited_cells
            for direction in ['top', 'right', 'bottom', 'left']:
                next = self.get_next_cell(current, direction)
                if next and next not in visited and self.can_move(current, next):
                    new_cost = current_cost + self.get_move_cost(current, next)
                    new_path = path + [next]
                    heapq.heappush(frontier, (new_cost, next, new_path))

    def get_next_cell(self, current, direction):
        """Get the next cell based on the direction"""
        x, y = current
        if direction == 'top' and x > 0 and x < 21:
            return (x-1, y)
        if direction == 'right' and y < self.maze.num_cols-1:
            return (x, y+1)
        if direction == 'bottom' and x < self.maze.num_rows-1:
            return (x+1, y)
        if direction == 'left' and y > 0 and y < 21:
            return (x, y-1)
        return None

    def can_move(self, current, next):
        """Check if movement is possible without crossing walls"""
        current_cell = self.maze.grid[current[0]][current[1]]
        next_cell = self.maze.grid[next[0]][next[1]]
        if next[0] < current[0] and current_cell.walls['top']: return False
        if next[0] > current[0] and current_cell.walls['bottom']: return False
        if next[1] < current[1] and current_cell.walls['left']: return False
        if next[1] > current[1] and current_cell.walls['right']: return False
        return True

    def get_move_cost(self, current, next):
        """Determine the cost of moving from current to next position"""
        if current[0] == next[0] or current[1] == next[1]:  
            return 0.9 if current[1] == next[1] else 1.1
        else:  
            return 1.1

class AstarSearch(Solver):      #A star search class
    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):
        logging.debug('Class Astarsearch called')
        self.name = "astarsearch"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):
        logging.debug("Class UniformCostSearch solve called")
        start = self.maze.entry_coor
        goal = self.maze.exit_coor
        frontier = []   #다음 방문해야 하는 목록 중 아직 방문하지 않은 목록
        heapq.heappush(frontier, (0, start, [start]))  # (cost, position, path)
        visited = set() #방문한 목록들의 집합.
        all_visited_cells = []  #모든 방문한 셀을 저장하는 리스트

        while frontier:
            current_cost, current, path = heapq.heappop(frontier)

            if current in visited:
                tt=(current, False)
                all_visited_cells.append(tt)
                continue
            visited.add(current)
            ff=(current, True)
            all_visited_cells.append(ff)    #output

            if current == goal:
                print("Path: {path}")
                print("Searched Cells: ", all_visited_cells)
                return all_visited_cells
            for direction in ['top', 'right', 'bottom', 'left']:
                next = self.get_next_cell(current, direction)
                if next and next not in visited and self.can_move(current, next):
                    new_cost = current_cost + self.get_move_cost(current, next)
                    new_path = path + [next]
                    heapq.heappush(frontier, (new_cost + self.heuristic(next, goal), next, new_path))

    def get_next_cell(self, current, direction):
        """Get the next cell based on the direction"""
        x, y = current
        if direction == 'top' and x > 0 and x < 21:
            return (x-1, y)
        if direction == 'right' and y < self.maze.num_cols-1:
            return (x, y+1)
        if direction == 'bottom' and x < self.maze.num_rows-1:
            return (x+1, y)
        if direction == 'left' and y > 0 and y < 21:
            return (x, y-1)
        return None

    def can_move(self, current, next):
        """Check if movement is possible without crossing walls"""
        current_cell = self.maze.grid[current[0]][current[1]]
        next_cell = self.maze.grid[next[0]][next[1]]
        if next[0] < current[0] and current_cell.walls['top']: return False
        if next[0] > current[0] and current_cell.walls['bottom']: return False
        if next[1] < current[1] and current_cell.walls['left']: return False
        if next[1] > current[1] and current_cell.walls['right']: return False
        return True

    def get_move_cost(self, current, next):
        """Determine the cost of moving from current to next position"""
        if current[0] == next[0] or current[1] == next[1]:  
            return 0.9 if current[1] == next[1] else 1.1
        else:  
            return 1.1
    def heuristic(self, a, b):
        """Calculate the Manhattan distance from a to b"""
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)