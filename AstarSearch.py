from __future__ import absolute_import
from src.maze_manager import MazeManager


if __name__ == "__main__":

    # Create the manager
    manager = MazeManager()

    # Add a 20x20 maze to the manager
    maze = manager.add_maze(20, 20)

    # Solve the maze using the Breadth First algorithm
    manager.show_maze(maze.id)
    manager.solve_maze(maze.id, "astarsearch")
    # Show how the maze was solved
    manager.show_solution_animation(maze.id)  #탐색한 list를 확인하고 싶다면 주석처리

