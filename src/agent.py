import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64

class Agent(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=4):
        super().__init__()
        self.fitness = 0.0
        self.position = None  # Current position in the maze
        self.best_position = None  # Best position reached so far
        self.steps_taken = 0  # Track number of steps
        self.reached_goal = False  # Flag for goal completion
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # More capable neural network with two hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Initialize with better weights for slightly better initial behavior
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        
    def forward(self, obs):
        # Process the observation
        current_pos = torch.tensor(obs['current_pos'], dtype=torch.float32)
        goal_pos = torch.tensor(obs['goal_pos'], dtype=torch.float32)
        walls = torch.tensor(obs['walls'], dtype=torch.float32)
        
        # Calculate additional features
        direction_to_goal = goal_pos - current_pos
        
        # Combine features into a single input vector
        state = torch.cat([
            current_pos,  # Current position (2)
            goal_pos,     # Goal position (2)
            walls,        # Wall configuration (4)
            direction_to_goal  # Direction to goal (2)
        ])
        
        return self.net(state)
    
    def reset(self, start_position):
        """Reset the agent to starting position"""
        self.position = start_position
        self.best_position = start_position
        self.steps_taken = 0
        self.reached_goal = False
        self.fitness = 0.0
        
    def move(self, action, maze):
        """Move the agent based on the action and maze constraints"""
        if self.position is None:
            return False
            
        # Convert action to direction
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        direction = directions[action]
        
        # Calculate new position
        new_position = (self.position[0] + direction[0], self.position[1] + direction[1])
        
        # Check if move is valid
        if maze.is_valid_move(self.position, new_position):
            self.position = new_position
            self.steps_taken += 1
            
            # Update best position based on goal distance
            if self.best_position is None or np.linalg.norm(np.array(new_position) - np.array(maze.goal)) < np.linalg.norm(np.array(self.best_position) - np.array(maze.goal)):
                self.best_position = new_position
                
            # Check for goal
            if self.position == maze.goal:
                self.reached_goal = True
            
            return True
        
        return False
    
    def get_genome(self):
        return [p.data.numpy() for p in self.parameters()]
    
    def set_genome(self, genome):
        for param, weights in zip(self.parameters(), genome):
            param.data = torch.tensor(weights, dtype=torch.float32)
            
    def calculate_fitness(self, maze):
        """Calculate fitness based on goal achievement, steps taken, and distance"""
        if self.reached_goal:
            # Incentivize shorter paths
            max_possible_steps = maze.size * maze.size
            step_efficiency = max(0, max_possible_steps - self.steps_taken) / max_possible_steps
            self.fitness = 1.0 + step_efficiency
        else:
            # Distance-based fitness
            distance = np.linalg.norm(np.array(self.position) - np.array(maze.goal))
            max_possible_distance = np.linalg.norm(np.array((0, 0)) - np.array((maze.size, maze.size)))
            distance_component = (max_possible_distance - distance) / max_possible_distance
            step_penalty = min(1.0, self.steps_taken / (maze.size * maze.size * 2))
            
            self.fitness = 0.1 + (0.9 * distance_component) - (0.2 * step_penalty)
            
        return self.fitness
        
    def visualize_network(self, save_path=None):
        """
        Visualize the neural network weights and biases as heatmaps.
        
        Args:
            save_path: Optional path to save the visualization image
        
        Returns:
            If save_path is None, returns a base64 encoded PNG image
            Otherwise, saves the image to the specified path
        """
        # Get all network parameters
        params = list(self.parameters())
        weights = [p.data.numpy() for p in params if len(p.shape) > 1]  # Weights are matrices
        biases = [p.data.numpy() for p in params if len(p.shape) == 1]  # Biases are vectors
        
        # Create a figure to hold all visualizations
        layer_count = len(weights)
        fig, axes = plt.subplots(layer_count, 2, figsize=(12, 4 * layer_count))
        if layer_count == 1:
            axes = [axes]  # Handle single layer case
            
        # Define input feature names for better visualization
        input_names = ['Pos_x', 'Pos_y', 'Goal_x', 'Goal_y', 
                      'Wall_up', 'Wall_down', 'Wall_left', 'Wall_right',
                      'Dir_goal_x', 'Dir_goal_y']
                      
        output_names = ['Up', 'Down', 'Left', 'Right']
        
        # Layer names
        layer_names = ['Input → Hidden1', 'Hidden1 → Hidden2', 'Hidden2 → Output']
        
        # Set a consistent colormap
        cmap = plt.cm.RdBu_r
        
        # For each layer, plot weights and biases
        for i, (w, b) in enumerate(zip(weights, biases)):
            # Plot weights
            ax_w = axes[i][0]
            im = ax_w.imshow(w, cmap=cmap, aspect='auto', interpolation='none')
            ax_w.set_title(f'Layer {i+1} Weights: {layer_names[i]}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_w)
            
            # Add row and column labels for the first and last layer
            if i == 0:  # Input → Hidden
                ax_w.set_yticks(range(self.hidden_size))
                ax_w.set_xticks(range(self.input_size))
                ax_w.set_xticklabels(input_names, rotation=45, ha='right')
            elif i == layer_count - 1:  # Hidden → Output
                ax_w.set_yticks(range(self.output_size))
                ax_w.set_yticklabels(output_names)
            
            # Plot biases
            ax_b = axes[i][1]
            im = ax_b.imshow(b.reshape(-1, 1), cmap=cmap, aspect=0.2, interpolation='none')
            ax_b.set_title(f'Layer {i+1} Biases')
            
            # Add colorbar
            plt.colorbar(im, ax=ax_b)
            
            # Add labels for output biases
            if i == layer_count - 1:
                ax_b.set_yticks(range(self.output_size))
                ax_b.set_yticklabels(output_names)
        
        # Add figure title with fitness information
        fig.suptitle(f'Neural Network Visualization - Fitness: {self.fitness:.3f}' + 
                    (f' (Reached Goal in {self.steps_taken} steps)' if self.reached_goal else ''),
                    fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or return as base64
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            return save_path
        else:
            # Return as base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150)
            plt.close(fig)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            return base64.b64encode(image_png).decode('utf-8')
            
    def save_to_file(self, filepath, include_metadata=True):
        """
        Save the agent to a file, including neural network and metadata
        
        Args:
            filepath: Path to save the agent
            include_metadata: Whether to include position and fitness data
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get network parameters
        params = self.get_genome()
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_params = [p.tolist() for p in params]
        
        # Create data to save
        data = {
            'network_params': serializable_params,
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            }
        }
        
        # Add metadata if requested
        if include_metadata:
            data['metadata'] = {
                'fitness': float(self.fitness),
                'position': self.position,
                'best_position': self.best_position,
                'steps_taken': self.steps_taken,
                'reached_goal': self.reached_goal
            }
            
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
        
    @classmethod
    def load_from_file(cls, filepath):
        """
        Load an agent from a file
        
        Args:
            filepath: Path to load the agent from
            
        Returns:
            Agent instance loaded from the file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Create agent with the right architecture
        arch = data.get('architecture', {})
        agent = cls(
            input_size=arch.get('input_size', 10),
            hidden_size=arch.get('hidden_size', 16),
            output_size=arch.get('output_size', 4)
        )
        
        # Convert lists back to numpy arrays
        params = [np.array(p, dtype=np.float32) for p in data['network_params']]
        
        # Set network parameters
        agent.set_genome(params)
        
        # Set metadata if available
        if 'metadata' in data:
            meta = data['metadata']
            agent.fitness = meta.get('fitness', 0.0)
            agent.position = tuple(meta.get('position')) if meta.get('position') else None
            agent.best_position = tuple(meta.get('best_position')) if meta.get('best_position') else None
            agent.steps_taken = meta.get('steps_taken', 0)
            agent.reached_goal = meta.get('reached_goal', False)
            
        return agent