import numpy as np
import random

class Maze:
    def __init__(self, size=10):
        """
        Create a maze with the specified size
        
        Args:
            size: The width/height of the maze (will be a square)
        """
        self.size = size
        # Ensure odd-sized maze for better path generation
        if size % 2 == 0:
            size += 1
            self.size = size
            
        # Start with all walls
        self.grid = np.ones((size, size))
        
        # Always create open corners for start and goal
        self.start = (1, 1)
        self.goal = (size-2, size-2)
        
        # Create a simpler maze to ensure paths
        self._generate_maze()
        
        # Make sure start and goal are open
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
    def _generate_maze(self):
        """Generate a more structured and solvable maze"""
        # Carve a direct path from start to goal
        path_x, path_y = self.start[0], self.start[1]
        goal_x, goal_y = self.goal
        
        # Carve a path along the border first (more structured)
        for i in range(path_x, self.size-2):
            self.grid[i, path_y] = 0  # Vertical path on left
            
        for j in range(path_y, self.size-2):
            self.grid[self.size-2, j] = 0  # Horizontal path on bottom
            
        # Create some random branches off the main path
        for _ in range(self.size):
            x = random.randint(1, self.size-2)
            y = random.randint(1, self.size-2)
            
            # Create a small open area around this point
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 1 <= nx < self.size-1 and 1 <= ny < self.size-1:
                        if random.random() < 0.7:  # 70% chance to open cell
                            self.grid[nx, ny] = 0
        
        # Ensure there's a clear path by opening up more cells near the borders
        for i in range(2, self.size-2):
            if random.random() < 0.6:
                self.grid[i, 2] = 0  # Near left border
            if random.random() < 0.6:
                self.grid[2, i] = 0  # Near top border
            if random.random() < 0.6:
                self.grid[i, self.size-3] = 0  # Near right border
            if random.random() < 0.6:
                self.grid[self.size-3, i] = 0  # Near bottom border
        
    def get_observation(self, position):
        """Get the agent's observation at the current position"""
        return {
            'current_pos': np.array(position, dtype=np.float32),
            'goal_pos': np.array(self.goal, dtype=np.float32),
            'walls': self._get_nearby_walls(position).astype(np.float32)
        }
    
    def _get_nearby_walls(self, pos):
        """Get the wall configuration in the 4 adjacent cells"""
        walls = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_i, new_j = pos[0] + di, pos[1] + dj
            # Check if the position is within bounds
            if 0 <= new_i < self.size and 0 <= new_j < self.size:
                walls.append(self.grid[new_i, new_j])
            else:
                walls.append(1)  # Out of bounds is considered a wall
        return np.array(walls)
        
    def is_valid_move(self, position, new_position):
        """Check if a move to new_position from position is valid"""
        # Check if within bounds
        if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
            return False
            
        # Check if the target cell is a wall
        if self.grid[new_position[0], new_position[1]] == 1:
            return False
            
        return True