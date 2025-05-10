import numpy as np
from enum import IntEnum

class CellType(IntEnum):
    """Types of cells in the maze."""
    WALL = 0
    EMPTY = 1
    SPAWN = 2
    GOAL = 3

class Maze:
    """Simple maze environment."""
    
    def __init__(self, width, height):
        """Initialize an empty maze."""
        self.width = width
        self.height = height
        self.grid = np.full((height, width), CellType.WALL, dtype=np.int8)
        self.spawn_pos = None
        self.goal_pos = None
        self.maze_type = "custom" # Default type
    
    def set_cell(self, x, y, cell_type):
        """Set the type of a cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = cell_type
            
            # Track special positions
            if cell_type == CellType.SPAWN:
                self.spawn_pos = (x, y)
            elif cell_type == CellType.GOAL:
                self.goal_pos = (x, y)
    
    def get_cell(self, x, y):
        """Get the type of a cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x]
        return CellType.WALL  # Outside the grid is considered a wall
    
    def is_walkable(self, x, y):
        """Check if a cell can be walked on."""
        return self.get_cell(x, y) in (CellType.EMPTY, CellType.SPAWN, CellType.GOAL)
    
    def create_l_maze(self):
        """Create a simple L-shaped maze."""
        self.grid.fill(CellType.WALL) # Reset grid
        mid_x = self.width // 2
        mid_y = self.height // 2
        for x in range(1, mid_x + 1):
            self.set_cell(x, mid_y, CellType.EMPTY)
        for y in range(1, mid_y + 1):
            self.set_cell(mid_x, y, CellType.EMPTY)
        self.set_cell(1, mid_y, CellType.SPAWN)
        self.set_cell(mid_x, 1, CellType.GOAL)
        self.maze_type = "L-Shape"
        return self

    def create_u_shape_maze(self):
        """Create a simple U-shaped maze."""
        self.grid.fill(CellType.WALL) # Reset grid
        
        # Define path coordinates for U-shape (relative to size)
        path_width = self.width - 4 # Leave border
        path_height = self.height - 4
        top_y = 2
        bottom_y = self.height - 3
        left_x = 2
        right_x = self.width - 3

        # Create bottom horizontal section
        for x in range(left_x, right_x + 1):
            self.set_cell(x, bottom_y, CellType.EMPTY)
        
        # Create left vertical section
        for y in range(top_y, bottom_y + 1):
            self.set_cell(left_x, y, CellType.EMPTY)
            
        # Create right vertical section
        for y in range(top_y, bottom_y + 1):
            self.set_cell(right_x, y, CellType.EMPTY)

        # Set spawn and goal at the ends of the U
        self.set_cell(left_x + 1, top_y, CellType.SPAWN) # Top-left opening
        # Move goal one cell right, directly onto the right vertical path
        self.set_cell(right_x, top_y, CellType.GOAL)  
        self.maze_type = "U-Shape Maze"
        return self

    def create_c_shape_maze(self, passages=1):
        """Create a more complex C-shaped spiral maze (Improved goal placement)."""
        self.grid.fill(CellType.WALL) # Start with walls

        # Carve out spiral path
        x, y = 1, 1 
        dx, dy = 1, 0 
        last_carved_x, last_carved_y = x, y # Track last valid carved position
        end_of_path_coords = None # Store the coordinate *just before* the spiral stopped carving
        
        layer = 1
        while True:
            side_len = 0
            if dx == 1: side_len = self.width - 1 - (layer * 2)  # Moving right
            elif dy == 1: side_len = self.height - 1 - (layer * 2) # Moving down
            elif dx == -1: side_len = self.width - 1 - (layer * 2) # Moving left
            elif dy == -1: side_len = self.height - 1 - (layer * 2) # Moving up
            side_len = max(0, side_len + passages) 
            
            if side_len <= 0: 
                end_of_path_coords = (last_carved_x, last_carved_y)
                break # Stop if spiral closes

            for i in range(side_len):
                # Store the position *before* carving the potential last cell
                if i == side_len - 1: 
                    end_of_path_coords = (x, y)
                    
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.grid[y, x] == CellType.WALL:
                         self.set_cell(x, y, CellType.EMPTY)
                         last_carved_x, last_carved_y = x, y 
                    x += dx
                    y += dy
                else:
                    end_of_path_coords = (last_carved_x, last_carved_y) # Use last valid if hit boundary
                    side_len = 0 
                    break
            if side_len == 0: 
                if end_of_path_coords is None: # Ensure it's set even if loop breaks early
                     end_of_path_coords = (last_carved_x, last_carved_y)
                break 
                 
            dx, dy = -dy, dx 
            if dy == -1: 
                layer += 1
                x += 1 
                y += 1
        
        # Set spawn near the outside
        self.set_cell(1, 1, CellType.SPAWN)
        
        # Set goal at the calculated end of the carved path
        goal_placed = False
        if end_of_path_coords and self.is_walkable(end_of_path_coords[0], end_of_path_coords[1]) and end_of_path_coords != (1, 1):
            self.set_cell(end_of_path_coords[0], end_of_path_coords[1], CellType.GOAL)
            goal_placed = True
            print(f"Placed C-Maze goal at end of path: {end_of_path_coords}")
        
        # Fallback if primary placement failed (should be less common now)
        if not goal_placed:
            print(f"Warning: C-Maze primary goal placement failed (target: {end_of_path_coords}). Using fallback.")
            center_x, center_y = self.width // 2, self.height // 2
            best_fallback_goal = None
            max_dist_sq = -1
            # Search grid for furthest walkable point from spawn
            for gy in range(1, self.height - 1):
                for gx in range(1, self.width - 1):
                    if self.is_walkable(gx, gy) and (gx, gy) != (1,1):
                        dist_sq = (gx - 1)**2 + (gy - 1)**2
                        if dist_sq > max_dist_sq:
                            max_dist_sq = dist_sq
                            best_fallback_goal = (gx, gy)
            
            if best_fallback_goal:
                 self.set_cell(best_fallback_goal[0], best_fallback_goal[1], CellType.GOAL)
                 goal_placed = True
                 print(f"Placed C-Maze goal via fallback: {best_fallback_goal}")

        if not goal_placed: # Final desperate fallback
             print("Warning: Could not place goal intelligently in C-Shape maze! Using default.")
             self.set_cell(self.width - 2, self.height - 2, CellType.GOAL)
             
        self.maze_type = "C-Shape Maze"
        return self
    
    def to_rgb_image(self, cell_size=10):
        """Convert the maze to an RGB image."""
        img = np.zeros((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8)
        
        # Define colors for cell types
        colors = {
            CellType.WALL: (0, 0, 0),        # Black
            CellType.EMPTY: (200, 200, 200), # Light gray
            CellType.SPAWN: (0, 255, 0),     # Green
            CellType.GOAL: (255, 215, 0),    # Gold
        }
        
        # Fill the image with cell colors
        for y in range(self.height):
            for x in range(self.width):
                cell_type = self.grid[y, x]
                color = colors[cell_type]
                
                # Fill the cell with its color
                img[y*cell_size:(y+1)*cell_size, 
                    x*cell_size:(x+1)*cell_size] = color
        
        return img 