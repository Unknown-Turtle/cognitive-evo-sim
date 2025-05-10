import os
import pickle
import time
import numpy as np
import neat

from src.maze import Maze
from src.agent import Agent
from src.utils import save_genome, load_genome, visualize_network

# --- Configuration Constants ---
DEFAULT_MAZE_WIDTH = 32
DEFAULT_MAZE_HEIGHT = 32
DEFAULT_STEPS_PER_GENERATION = 100
DEFAULT_CHECKPOINT_INTERVAL = 5
DEFAULT_NEAT_CONFIG_PATH = os.path.join('configs', 'neat_config.txt')
DEFAULT_NEAT_TEMPLATE_PATH = os.path.join('configs', 'neat_config.template')
# -----------------------------

class MazeEvolution:
    """Handles the evolution of agents using NEAT."""
    
    def __init__(self, config_path=None, initial_maze_type="L-Shape"):
        """Initialize the evolution with a NEAT configuration and initial maze."""
        if config_path is None:
            config_path = DEFAULT_NEAT_CONFIG_PATH
            
            # If the config file doesn't exist, create it from the template
            if not os.path.exists(config_path):
                self.create_config_from_template()
        
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create the population
        self.population = neat.Population(self.config)
        
        # Add reporters to track progress
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        
        # List to store agents for visualization - Initialize agents *before* setting the maze
        self.agents = []
        
        # Create the initial maze
        self.maze = Maze(DEFAULT_MAZE_WIDTH, DEFAULT_MAZE_HEIGHT) # Use constants
        self.set_maze(initial_maze_type) # Populate with the specified type
        
        # Set up evolution parameters
        self.generation = 0
        self.steps_per_generation = DEFAULT_STEPS_PER_GENERATION # Use constant
        self.best_fitness = 0
        self.best_genome = None
        
        # Store historical data for visualization
        self.fitness_history = {
            'generations': [],
            'best_fitnesses': [],
            'avg_fitnesses': [],
            'worst_fitnesses': []
        }
        
        # Store species data (even if not visualized currently)
        self.species_history = {
            'generations': [],
            'species_sizes': []
        }
        
        # Store neural network complexity data
        self.complexity_history = []
        
        # Store solution diversity data (weights)
        self.diversity_history = []
        
        # Store agent paths
        self.agent_paths = []
        
        # Path tracking flag - disable by default since we don't use the path visualization
        self.path_tracking_enabled = False
        
        # Checkpointing settings
        self.checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL # Use constant
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # current_maze_type is set within self.set_maze()
    
    def create_config_from_template(self):
        """Create a NEAT configuration file from the template file."""
        config_dir = os.path.dirname(DEFAULT_NEAT_CONFIG_PATH)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        # Default template content if the template file doesn't exist
        default_template_content = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 15.0
pop_size              = 150
reset_on_extinction   = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 4.0
bias_min_value          = -4.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = partial_direct 0.5
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 12
num_inputs              = 8
num_outputs             = 5
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 4.0
response_min_value      = -4.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 4.0
weight_min_value        = -4.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        
        template_content = default_template_content
        try:
            if os.path.exists(DEFAULT_NEAT_TEMPLATE_PATH):
                with open(DEFAULT_NEAT_TEMPLATE_PATH, 'r') as f_template:
                    template_content = f_template.read()
            else:
                # Create the template file if it doesn't exist
                with open(DEFAULT_NEAT_TEMPLATE_PATH, 'w') as f_template:
                    f_template.write(default_template_content)
                print(f"Created default NEAT config template at: {DEFAULT_NEAT_TEMPLATE_PATH}")

        except IOError as e:
            print(f"Warning: Could not read/write NEAT template file at {DEFAULT_NEAT_TEMPLATE_PATH}. Using hardcoded defaults. Error: {e}")
            template_content = default_template_content # Fallback
            
        try:
            with open(DEFAULT_NEAT_CONFIG_PATH, 'w') as f_config:
                f_config.write(template_content)
            print(f"Created NEAT config from template: {DEFAULT_NEAT_CONFIG_PATH}")
            return DEFAULT_NEAT_CONFIG_PATH
        except IOError as e:
            print(f"Error writing NEAT config file to {DEFAULT_NEAT_CONFIG_PATH}: {e}")
            return None

    def evaluate_genomes(self, genomes, config):
        """Evaluate a list of genomes."""
        self.agents = []
        
        # Create agents for each genome
        for genome_id, genome in genomes:
            # Reset genome fitness
            genome.fitness = 0.0
            
            # Create an agent with this genome
            agent = Agent(genome, config, *self.maze.spawn_pos)
            self.agents.append((genome_id, agent))
            
            # Initialize path tracking if enabled
            if self.path_tracking_enabled:
                agent.path = [(agent.x, agent.y)]
        
        # Run simulation for each agent
        for _ in range(self.steps_per_generation):
            # Move each agent
            for genome_id, agent in self.agents:
                if not agent.reached_goal:
                    agent.move(self.maze)
                    
                    # Track path if enabled
                    if self.path_tracking_enabled:
                        agent.path.append((agent.x, agent.y))
        
        # Calculate fitness for each agent
        fitness_values = []
        for genome_id, agent in self.agents:
            # Calculate the fitness
            fitness = agent.calculate_fitness(self.steps_per_generation)
            fitness_values.append(fitness)
            
            # Find the corresponding genome and set its fitness
            for gid, genome in genomes:
                if gid == genome_id:
                    genome.fitness = fitness
                    
                    # Update best genome if this is the best so far
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_genome = genome
                    break
        
        # Store paths for heatmap if tracking is enabled
        if self.path_tracking_enabled:
            self.agent_paths = [agent.path for _, agent in self.agents]
    
    def get_agent_paths_data(self):
        """Get the agent paths data for heatmap visualization."""
        return self.agent_paths, self.maze 
        
    def save_checkpoint(self, custom_filename=None):
        """Save a checkpoint of the current evolutionary state.
        
        Args:
            custom_filename: Optional custom filename for the checkpoint
            
        Returns:
            str: Path to the saved checkpoint file
        """
        try:
            # Create checkpoint directory if it doesn't exist
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            
            # Generate checkpoint filename based on generation or custom name
            if custom_filename:
                checkpoint_filename = f"{custom_filename}.checkpoint"
            else:
                checkpoint_filename = f"generation_{self.generation}.checkpoint"
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'generation': self.generation,
                'population': self.population,
                'best_fitness': self.best_fitness,
                'best_genome': self.best_genome,
                'fitness_history': self.fitness_history,
                'species_history': self.species_history,
                'complexity_history': self.complexity_history, # Save complexity history
                'diversity_history': self.diversity_history,   # Save diversity history
                'config': self.config,
                'steps_per_generation': self.steps_per_generation,
                'path_tracking_enabled': self.path_tracking_enabled,
                'current_maze_type': self.current_maze_type, # Save current maze type
                'timestamp': time.time(),
                'agent_paths': self.agent_paths, # Still save paths, even if disabled for viz
                # 'maze': self.maze # Saving the whole maze object might be fragile, save type instead
            }
            
            # Save checkpoint data
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"Checkpoint saved to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path):
        """Load evolution state from a checkpoint file."""
        try:
            # Check if checkpoint file exists
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            # Load checkpoint data
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore evolution state
            self.generation = checkpoint_data['generation']
            self.population = checkpoint_data['population']
            self.best_fitness = checkpoint_data['best_fitness']
            self.best_genome = checkpoint_data['best_genome']
            self.fitness_history = checkpoint_data['fitness_history']
            self.species_history = checkpoint_data['species_history']
            self.complexity_history = checkpoint_data.get('complexity_history', []) # Load complexity history
            self.diversity_history = checkpoint_data.get('diversity_history', [])   # Load diversity history
            self.config = checkpoint_data['config']
            self.steps_per_generation = checkpoint_data['steps_per_generation']
            self.path_tracking_enabled = checkpoint_data['path_tracking_enabled']
            
            # Restore maze type and set the maze
            loaded_maze_type = checkpoint_data.get('current_maze_type', 'L-Shape') # Default to L-Shape
            if not hasattr(self, 'maze'): # Create maze object if it doesn't exist
                self.maze = Maze(32, 32)
            self.set_maze(loaded_maze_type)
                
            # Restore agent paths if available
            self.agent_paths = checkpoint_data.get('agent_paths', [])
            
            # Re-add reporters to the population
            self.population.reporters.reporters = [] # Clear old reporters first
            self.population.add_reporter(neat.StdOutReporter(True))
            self.stats = neat.StatisticsReporter()
            self.population.add_reporter(self.stats)
            
            # Reconstruct agents from the saved population for visualization
            self._reconstruct_agents_from_checkpoint()
            
            print(f"Checkpoint loaded from {checkpoint_path} (Generation {self.generation}, Maze: {self.current_maze_type})")
            return True
            
        except FileNotFoundError:
             print(f"Error: Checkpoint file not found at {checkpoint_path}")
             return False
        except (pickle.PickleError, EOFError) as e:
             print(f"Error loading checkpoint file (corrupted?): {e}")
             return False
        except (KeyError, AttributeError) as e:
             print(f"Error: Checkpoint data missing expected key or attribute: {e}")
             return False
        except Exception as e: # Catch other potential errors
            print(f"An unexpected error occurred loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_available_checkpoints(self):
        """Get a list of available checkpoint files.
        
        Returns:
            list: List of tuples (filename, generation, timestamp)
        """
        checkpoints = []
        
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith('.checkpoint'):
                    try:
                        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                        with open(checkpoint_path, 'rb') as f:
                            checkpoint_data = pickle.load(f)
                        
                        generation = checkpoint_data.get('generation', 0)
                        timestamp = checkpoint_data.get('timestamp', 0)
                        
                        checkpoints.append((filename, generation, timestamp))
                    except:
                        # Skip corrupted files
                        pass
        
        # Sort by generation
        checkpoints.sort(key=lambda x: x[1])
        return checkpoints
    
    def auto_checkpoint(self):
        """Automatically save a checkpoint if the current generation is a multiple of the checkpoint interval."""
        if self.generation % self.checkpoint_interval == 0 and self.generation > 0:
            self.save_checkpoint()
    
    def _update_histories(self):
        """Update all historical tracking data after a generation."""
        # Fitness History
        if self.agents: # Check if agents exist (might not on first call or error)
            fitness_values = [agent.calculate_fitness(self.steps_per_generation) for _, agent in self.agents]
            avg_fitness = np.mean(fitness_values) if fitness_values else 0
            worst_fitness = min(fitness_values) if fitness_values else 0
        else:
            avg_fitness = 0
            worst_fitness = 0
            
        self.fitness_history['generations'].append(self.generation)
        self.fitness_history['best_fitnesses'].append(self.best_fitness)
        self.fitness_history['avg_fitnesses'].append(avg_fitness)
        self.fitness_history['worst_fitnesses'].append(worst_fitness)
        
        # Species History
        try:
            species_sizes = self.stats.get_species_sizes()
            self.species_history['generations'].append(self.generation)
            self.species_history['species_sizes'].append(species_sizes)
        except Exception:
            pass # Ignore if stats reporter hasn't run yet
            
        # Network Complexity History
        self.update_complexity_history()
        
        # Solution Diversity History
        self.update_diversity_history()
        
    def _reconstruct_agents_from_checkpoint(self):
        """Helper to reconstruct agent objects after loading population from checkpoint."""
        self.agents = []
        if not self.population or not self.population.population:
            print("Warning: Population data missing or empty in checkpoint. Cannot reconstruct agents.")
            return
            
        genomes_dict = self.population.population
        for genome_id, genome in genomes_dict.items():
             # Ensure fitness is loaded into the genome object
            if genome.fitness is None and hasattr(self.population.reporters.reporters[-1], 'most_fit_genomes'):
                # Attempt to recover fitness if possible from stats reporter (may not always work)
                # This is a fallback, ideally fitness is saved with the genome by neat-python
                pass # Placeholder - neat should handle fitness persistence within population object

            agent = self.create_agent_from_genome(genome)
            # Agent position is reset by set_maze called during checkpoint load
            self.agents.append((genome_id, agent))
        print(f"Reconstructed {len(self.agents)} agents from checkpoint population.")

    def run_generation(self):
        """Run one generation of evolution."""
        if self.generation == 0:
            # For the first generation, evaluate the initial population
            genomes = list(self.population.population.items())
            self.evaluate_genomes([(genome_id, genome) for genome_id, genome in genomes], self.config)
            self.generation += 1
        else:
            # For subsequent generations, run NEAT
            self.population.run(self.evaluate_genomes, 1)
            self.generation += 1
        
        # Update tracking data
        self._update_histories()
                
        # Get species count for stats display
        species_count = 0
        try:
            # Access species data carefully
            species_set = self.population.species
            if species_set and species_set.species:
                 species_count = len(species_set.species)
        except Exception as e:
             print(f"Warning: Could not get species count: {e}")
             species_count = 0 # Fallback
        
        # Auto-checkpoint if needed
        self.auto_checkpoint()
                    
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.fitness_history['avg_fitnesses'][-1] if self.fitness_history['avg_fitnesses'] else 0,
            'species_count': species_count,
            'completion_rate': sum(1 for _, agent in self.agents if agent.reached_goal) / len(self.agents) if self.agents else 0
        }
    
    def update_complexity_history(self):
        """Update the network complexity history with current generation data."""
        if not self.population.population:
            return  # No data available yet
            
        total_nodes = 0
        total_connections = 0
        total_enabled_connections = 0
        
        # Count nodes and connections across all genomes
        for genome_id, genome in self.population.population.items():
            # Count nodes (input, output, hidden)
            total_nodes += len(genome.nodes)
            
            # Count connections (total and enabled)
            for conn in genome.connections.values():
                total_connections += 1
                if conn.enabled:
                    total_enabled_connections += 1
        
        # Calculate average per genome
        num_genomes = len(self.population.population)
        avg_nodes = total_nodes / num_genomes if num_genomes > 0 else 0
        avg_connections = total_connections / num_genomes if num_genomes > 0 else 0
        avg_enabled_connections = total_enabled_connections / num_genomes if num_genomes > 0 else 0
        
        # Store the data
        self.complexity_history.append((
            self.generation,
            avg_nodes,
            avg_connections,
            avg_enabled_connections
        ))
    
    def get_complexity_history_data(self):
        """Get the network complexity history data for visualization."""
        return self.complexity_history
    
    def get_best_agents(self, n=5):
        """Get the top n performing agents."""
        # Sort agents by fitness
        sorted_agents = sorted(
            [(genome_id, agent) for genome_id, agent in self.agents],
            key=lambda x: x[1].calculate_fitness(self.steps_per_generation),
            reverse=True
        )
        
        # Return the top n
        return [agent for _, agent in sorted_agents[:n]]
    
    def save_best_genome(self, filename=None):
        """Save the best genome."""
        if self.best_genome:
            return save_genome(self.best_genome, self.config, filename, self.generation)
        return None, None
    
    def load_genome(self, filename='best_maze_genome.pickle'):
        """Load a genome from a file."""
        return load_genome(filename)
    
    def create_agent_from_genome(self, genome):
        """Create an agent with the given genome."""
        return Agent(genome, self.config, *self.maze.spawn_pos)
    
    def enable_path_tracking(self, enabled=True):
        """Enable or disable agent path tracking for heatmap visualization."""
        self.path_tracking_enabled = enabled
        
        # Clear existing paths when disabling
        if not enabled:
            self.agent_paths = []
    
    def get_fitness_history_data(self):
        """Get the fitness history data for visualization."""
        return self.fitness_history
        
    def get_species_history_data(self):
        """Get the species history data for visualization."""
        return self.species_history
    
    def update_diversity_history(self):
        """Update the solution diversity history with network weights and fitness from the current population."""
        if not self.population.population:
            return

        genome_data = []
        for genome_id, genome in self.population.population.items():
            weights = []
            # Extract weights from enabled connections
            for cg in genome.connections.values():
                if cg.enabled:
                    weights.append(cg.weight)
            # Also include node biases
            for ng in genome.nodes.values():
                weights.append(ng.bias)
                weights.append(ng.response) # Include response if used
                
            if weights: # Only include genomes with weights
                # Ensure fitness is assigned (it should be from evaluate_genomes)
                fitness = genome.fitness if genome.fitness is not None else 0.0 
                genome_data.append((np.array(weights), fitness))
        
        if genome_data:
            self.diversity_history.append((self.generation, genome_data))

    def get_diversity_history_data(self):
        """Get the solution diversity history data (weights and fitness) for visualization."""
        return self.diversity_history

    def set_maze(self, maze_type):
        """Set the current maze to a new type and reset agent positions."""
        print(f"Switching maze to: {maze_type}")
        if maze_type == "L-Shape":
            self.maze.create_l_maze()
        elif maze_type == "U-Shape Maze": # Renamed from Easy Maze
            self.maze.create_u_shape_maze() # Renamed method call
        elif maze_type == "C-Shape Maze": # Renamed from Hard Spiral
            self.maze.create_c_shape_maze() # Renamed method call
        else:
            print(f"Warning: Unknown maze type '{maze_type}'. Using L-Shape.")
            self.maze.create_l_maze()
        
        # Reset existing agent positions to the new spawn point
        # This does NOT reset their genomes or evolutionary progress
        if self.agents:
            print(f"Resetting positions of {len(self.agents)} agents to spawn {self.maze.spawn_pos}")
            for genome_id, agent in self.agents:
                agent.reset_position(*self.maze.spawn_pos)
        
        # Also clear any old path data if tracking was enabled
        self.agent_paths = [] 
        
        # Update maze type stored in the evolution object (useful for checkpointing/UI)
        self.current_maze_type = self.maze.maze_type 