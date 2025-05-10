import random
import neat

from src.maze import CellType

class Agent:
    """Neural network agent that navigates the maze."""
    
    def __init__(self, genome, config, x, y):
        """Initialize an agent with a neural network.
        
        Args:
            genome: NEAT genome
            config: NEAT config
            x: Initial x position
            y: Initial y position
        """
        # Create neural network from genome
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Set initial position
        self.x = x
        self.initial_x = x
        self.y = y
        self.initial_y = y
        
        # Track goal reaching
        self.reached_goal = False
        self.steps_taken = 0
        
        # Track path for visualization
        self.path = []
        
        self.distance_to_goal = float('inf')
        self.closest_approach = float('inf')
    
    def reset(self):
        """Reset the agent to its starting position."""
        self.x = self.initial_x
        self.initial_y = self.initial_y
        self.steps_taken = 0
        self.reached_goal = False
        self.distance_to_goal = float('inf')
        self.closest_approach = float('inf')
        self.path = [] # Clear path history
    
    def reset_position(self, x, y):
        """Reset the agent to a new position (e.g., when maze changes)."""
        self.x = x
        self.initial_x = x
        self.y = y
        self.initial_y = y
        self.steps_taken = 0
        self.reached_goal = False
        self.distance_to_goal = float('inf')
        self.closest_approach = float('inf')
        self.path = [] # Clear path history
        print(f"Agent position reset to {x}, {y}") # Debug print
    
    def sense(self, maze):
        """Get sensor inputs from the environment."""
        # Detect walls in four directions (N, E, S, W)
        wall_sensors = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            # 1.0 if there's a wall, 0.0 if not
            wall_sensors.append(1.0 if maze.get_cell(nx, ny) == CellType.WALL else 0.0)
        
        # Get direction to goal
        goal_direction = [0.0, 0.0, 0.0, 0.0]  # N, E, S, W
        if maze.goal_pos:
            gx, gy = maze.goal_pos
            # Calculate normalized direction vector to goal
            dx = gx - self.x
            dy = gy - self.y
            
            # Update distance to goal
            distance = max(abs(dx), abs(dy))  # Manhattan distance
            self.distance_to_goal = distance
            self.closest_approach = min(self.closest_approach, distance)
            
            # Set direction neurons
            if dy < 0:  # Goal is to the north
                goal_direction[0] = 1.0
            if dx > 0:  # Goal is to the east
                goal_direction[1] = 1.0
            if dy > 0:  # Goal is to the south
                goal_direction[2] = 1.0
            if dx < 0:  # Goal is to the west
                goal_direction[3] = 1.0
        
        # Combine sensor inputs
        return wall_sensors + goal_direction
    
    def think(self, inputs):
        """Use the neural network to decide on an action."""
        # Get outputs from neural network
        outputs = self.network.activate(inputs)
        
        # Determine direction of movement based on highest output
        # Outputs are [north, east, south, west, random]
        if outputs[4] > 0.5:  # Random movement
            return random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])
        else:
            # Find direction with highest activation
            max_index = outputs.index(max(outputs[:4]))
            if max_index == 0:  # North
                return (0, -1)
            elif max_index == 1:  # East
                return (1, 0)
            elif max_index == 2:  # South
                return (0, 1)
            else:  # West
                return (-1, 0)
    
    def move(self, maze):
        """Sense the environment, think, and move."""
        # Increment step counter
        self.steps_taken += 1
        
        # Check if already at goal
        if self.x == maze.goal_pos[0] and self.y == maze.goal_pos[1]:
            self.reached_goal = True
            return
        
        # Get sensor inputs
        inputs = self.sense(maze)
        
        # Get movement decision
        dx, dy = self.think(inputs)
        
        # Check if the move is valid
        new_x, new_y = self.x + dx, self.y + dy
        if maze.is_walkable(new_x, new_y):
            self.x, self.y = new_x, new_y
            
            # Check if reached goal
            if (self.x, self.y) == maze.goal_pos:
                self.reached_goal = True
    
    def calculate_fitness(self, max_steps):
        """Calculate the fitness of this agent."""
        # Base fitness from closest approach (0 to 1)
        if self.closest_approach == float('inf') or self.closest_approach == 0:
            base_fitness = 0.0
        else:
            base_fitness = 1.0 / self.closest_approach
        
        # Bonus for reaching goal
        goal_bonus = 10.0 if self.reached_goal else 0.0
        
        # Efficiency bonus for reaching goal quickly
        efficiency = 0.0
        if self.reached_goal:
            efficiency = (max_steps - self.steps_taken) / max_steps
        
        # Combined fitness
        return base_fitness + goal_bonus + (efficiency * 5.0) 