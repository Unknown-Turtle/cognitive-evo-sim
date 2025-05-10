import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

def save_genome(genome, config, filename=None, generation=None):
    """Save a genome to a file and visualize its neural network.
    
    Args:
        genome: The genome to save
        config: NEAT configuration object
        filename: Optional filename to use
        generation: Current generation number (optional)
        
    Returns:
        tuple: (path to saved genome, path to visualization)
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create neural network map directory if it doesn't exist
    neural_network_dir = os.path.join(os.getcwd(), 'data', 'neural_network_map')
    if not os.path.exists(neural_network_dir):
        os.makedirs(neural_network_dir)
    
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save genome
    if filename:
        genome_path = os.path.join(logs_dir, filename)
    else:
        genome_path = os.path.join(logs_dir, f'best_genome_{timestamp}.pickle')
    
    with open(genome_path, 'wb') as f:
        pickle.dump(genome, f)
    
    # Generate neural network visualization
    vis_path = os.path.join(neural_network_dir, f'network_{timestamp}.png')
    visualize_network(genome, config, vis_path, generation)
    
    return genome_path, vis_path

def load_genome(filename):
    """Load a genome from a file.
    
    Args:
        filename: Path to the genome file
        
    Returns:
        The loaded genome or None if loading failed
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                genome = pickle.load(f)
            return genome
        except Exception as e:
            print(f"Error loading genome: {e}")
    return None

def visualize_network(genome, config, output_path, generation=None):
    """Visualize the neural network structure using matplotlib.
    
    Args:
        genome: The genome to visualize
        config: The NEAT configuration
        output_path: Path to save the visualization image
        generation: Generation number (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import neat
        
        # Get timestamp for the title
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get network information
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Get all nodes (input, hidden, output)
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [node for node in genome.nodes.keys() 
                       if node not in input_nodes and node not in output_nodes]
        
        # Set up node positions
        node_positions = {}
        
        # Input nodes at the bottom
        input_x = 0.1
        input_spacing = 0.8 / (len(input_nodes) + 1)
        for i, node in enumerate(input_nodes):
            node_positions[node] = (input_x + (i + 1) * input_spacing, 0.1)
        
        # Output nodes at the top
        output_x = 0.1
        output_spacing = 0.8 / (len(output_nodes) + 1)
        for i, node in enumerate(output_nodes):
            node_positions[node] = (output_x + (i + 1) * output_spacing, 0.9)
        
        # Hidden nodes in the middle
        if hidden_nodes:
            hidden_layers = 1  # Simple layout with one hidden layer
            hidden_y = 0.5     # Middle of the plot
            hidden_spacing = 0.8 / (len(hidden_nodes) + 1)
            for i, node in enumerate(hidden_nodes):
                node_positions[node] = (hidden_spacing * (i + 1) + 0.1, hidden_y)
        
        # Set up the plot
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw connections
        for connection in genome.connections.values():
            if connection.enabled:
                from_node, to_node = connection.key
                weight = connection.weight
                
                # Get node positions
                start_x, start_y = node_positions[from_node]
                end_x, end_y = node_positions[to_node]
                
                # Calculate color based on weight
                color = 'green' if weight > 0 else 'red'
                alpha = min(abs(weight) / 2, 1.0)  # Transparency based on weight
                width = 0.5 + abs(weight) / 2.0    # Width based on weight
                
                # Draw the connection
                plt.plot([start_x, end_x], [start_y, end_y], 
                         color=color, alpha=alpha, linewidth=width)
        
        # Draw nodes
        node_colors = {
            'input': 'lightblue',
            'output': 'lightgreen',
            'hidden': 'lightgray'
        }
        
        # Node labels
        node_labels = {
            -1: 'N-dir', -2: 'E-dir', -3: 'S-dir', -4: 'W-dir',
            -5: 'N-wall', -6: 'E-wall', -7: 'S-wall', -8: 'W-wall',
            0: 'N-move', 1: 'E-move', 2: 'S-move', 3: 'W-move', 4: 'Random'
        }
        
        # Draw input nodes
        for node in input_nodes:
            x, y = node_positions[node]
            plt.scatter(x, y, s=200, color=node_colors['input'], zorder=10)
            label = node_labels.get(node, str(node))
            plt.text(x, y - 0.05, label, ha='center', va='center', fontsize=8)
        
        # Draw output nodes
        for node in output_nodes:
            x, y = node_positions[node]
            plt.scatter(x, y, s=200, color=node_colors['output'], zorder=10)
            label = node_labels.get(node, str(node))
            plt.text(x, y + 0.05, label, ha='center', va='center', fontsize=8)
        
        # Draw hidden nodes
        for i, node in enumerate(hidden_nodes):
            x, y = node_positions[node]
            plt.scatter(x, y, s=150, color=node_colors['hidden'], zorder=10)
            label = f"n{i}"  # Label hidden nodes as n0, n1, n2, etc.
            plt.text(x, y, label, ha='center', va='center', fontsize=7)
        
        # Add title and legend
        title = "Neural Network Structure"
        if generation is not None:
            title = f"{title} - Generation {generation}"
        plt.title(f"{title}\nNodes: {len(node_positions)} | Connections: {len(genome.connections)}\n{timestamp}")
        
        # Add legend for connection weights
        plt.plot([0.8, 0.9], [0.05, 0.05], color='green', linewidth=2, label='Positive Weight')
        plt.plot([0.8, 0.9], [0.02, 0.02], color='red', linewidth=2, label='Negative Weight')
        plt.legend(loc='lower right', fontsize=8)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating neural network visualization: {e}")
        return False

def plot_fitness_history(stats, output_path):
    """Create a plot of fitness over generations.
    
    Args:
        stats: Statistics dict with generations, best_fitnesses, avg_fitnesses, worst_fitnesses
        output_path: Path to save the visualization image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate timestamp for the title
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract data
        generations = stats['generations']
        best_fitnesses = stats['best_fitnesses']
        avg_fitnesses = stats['avg_fitnesses']
        if 'worst_fitnesses' in stats:
            worst_fitnesses = stats['worst_fitnesses']
            has_worst = True
        else:
            has_worst = False
            
        # Get current generation
        current_gen = generations[-1] if generations else 0

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitnesses, 'r-', label='Best Fitness')
        plt.plot(generations, avg_fitnesses, 'b-', label='Average Fitness')
        if has_worst:
            plt.plot(generations, worst_fitnesses, 'g-', label='Worst Fitness')
        
        # Add labels and grid
        plt.title(f'Fitness History - Generation {current_gen}\n{timestamp}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
    except (ValueError, TypeError) as e:
        print(f"Error processing data for fitness history plot: {e}")
        return False
    except IOError as e:
        print(f"Error saving fitness history plot to {output_path}: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"Unexpected error generating fitness history plot: {e}")
        traceback.print_exc()
        return False

def plot_species_diversity(stats, output_path):
    """Create a stacked area chart showing species diversity over generations.
    
    Args:
        stats: Statistics dict with generations and species_sizes
        output_path: Path to save the visualization image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate timestamp for the title
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract data
        generations = stats['generations']
        species_sizes = stats['species_sizes']
        
        # Get current generation
        current_gen = generations[-1] if generations else 0
        
        # Create a line chart instead of stackplot for more compatibility
        plt.figure(figsize=(10, 6))
        
        # Get unique species IDs across all generations
        all_species = set()
        for gen_data in species_sizes:
            for species_id in gen_data.keys():
                all_species.add(species_id)
        
        # Sort species IDs for consistent colors
        all_species = sorted(list(all_species))
        
        # Create a consistent color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_species)))
        
        # Plot line for each species
        legend_handles = []
        for i, species_id in enumerate(all_species):
            # Initialize data for this species
            y_data = []
            
            # Extract data for all generations
            for gen_data in species_sizes:
                if isinstance(gen_data, dict):
                    # Handle dict format (species_id → size)
                    y_data.append(gen_data.get(species_id, 0))
                elif isinstance(gen_data, list):
                    # Handle list format (index → size)
                    if i < len(gen_data):
                        y_data.append(gen_data[i])
                    else:
                        y_data.append(0)
                else:
                    # Unknown format, skip
                    continue
            
            # Plot this species as a line
            line, = plt.plot(generations, y_data, '-', 
                            color=colors[i], 
                            linewidth=2,
                            label=f'Species {species_id}')
            legend_handles.append(line)
        
        # Fill between lines for area effect if desired
        if legend_handles:
            plt.stackplot(generations, 
                        [plt.getp(h, 'ydata') for h in legend_handles],
                        colors=[plt.getp(h, 'color') for h in legend_handles],
                        alpha=0.3)
        
        # Add labels
        plt.title(f'Species Diversity - Generation {current_gen}\n{timestamp}')
        plt.xlabel('Generation')
        plt.ylabel('Population Size')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend if there aren't too many species
        if len(all_species) <= 10:
            plt.legend(legend_handles, [f'Species {s}' for s in all_species], 
                      loc='upper right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
    except Exception as e:
        import traceback
        print(f"Error generating species diversity plot: {e}")
        traceback.print_exc()  # Print the full stack trace for debugging
        return False

def create_agent_heatmap(maze, paths, output_path):
    """Create a heatmap showing the most visited cells in the maze.
    
    Args:
        maze: The maze object
        paths: List of paths, where each path is a list of (x, y) positions
        output_path: Path to save the visualization image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a 2D numpy array to store visit counts
        heatmap = np.zeros((maze.height, maze.width))
        
        # Count visits to each cell
        for path in paths:
            for x, y in path:
                # Make sure coordinates are within bounds
                if 0 <= x < maze.width and 0 <= y < maze.height:
                    heatmap[y, x] += 1
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Display the maze as the background
        maze_array = np.zeros((maze.height, maze.width, 3))
        for y in range(maze.height):
            for x in range(maze.width):
                cell_type = maze.get_cell(x, y)
                if cell_type == 1:  # Wall
                    maze_array[y, x] = [0, 0, 0]  # Black
                elif cell_type == 0:  # Empty
                    maze_array[y, x] = [0.9, 0.9, 0.9]  # Light gray
                elif cell_type == 2:  # Spawn
                    maze_array[y, x] = [0, 1, 0]  # Green
                elif cell_type == 3:  # Goal
                    maze_array[y, x] = [1, 0.84, 0]  # Gold
                    
        plt.imshow(maze_array, interpolation='nearest')
        
        # Apply heatmap on top with transparency
        # Mask out walls - don't show heatmap on walls
        wall_mask = np.zeros((maze.height, maze.width), dtype=bool)
        for y in range(maze.height):
            for x in range(maze.width):
                if maze.get_cell(x, y) == 1:  # Wall
                    wall_mask[y, x] = True
                    heatmap[y, x] = 0  # No visits to walls
        
        # Display heatmap with masked array
        plt.imshow(np.ma.array(heatmap, mask=wall_mask), 
                   cmap='hot', alpha=0.7, interpolation='nearest')
        
        # Add color bar
        cbar = plt.colorbar()
        cbar.set_label('Visit Count')
        
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add title
        plt.title('Agent Path Heatmap')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating agent heatmap: {e}")
        return False

def plot_network_complexity(generation_data, output_path):
    """Create a plot showing neural network complexity evolution over generations.
    
    Args:
        generation_data: List of tuples (generation, nodes, connections, enabled_connections)
        output_path: Path to save the visualization image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate timestamp for the title
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract data
        generations = [data[0] for data in generation_data]
        total_nodes = [data[1] for data in generation_data]
        total_connections = [data[2] for data in generation_data]
        enabled_connections = [data[3] for data in generation_data]
        
        # Get current generation
        current_gen = generations[-1] if generations else 0

        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot total nodes on left y-axis
        color_nodes = 'tab:blue'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Nodes', color=color_nodes)
        ax1.plot(generations, total_nodes, 'o-', color=color_nodes, label='Total Nodes')
        ax1.tick_params(axis='y', labelcolor=color_nodes)
        
        # Create second y-axis for connections
        ax2 = ax1.twinx()
        color_connections = 'tab:red'
        ax2.set_ylabel('Number of Connections', color=color_connections)
        ax2.plot(generations, total_connections, 's-', color=color_connections, label='Total Connections')
        ax2.plot(generations, enabled_connections, '^-', color='tab:green', label='Enabled Connections')
        ax2.tick_params(axis='y', labelcolor=color_connections)
        
        # Add title
        plt.title(f'Neural Network Complexity - Generation {current_gen}\n{timestamp}')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
    except (IndexError, ValueError, TypeError) as e: # Added IndexError for empty lists
        print(f"Error processing data for network complexity plot: {e}")
        return False
    except IOError as e:
        print(f"Error saving network complexity plot to {output_path}: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"Unexpected error generating network complexity plot: {e}")
        traceback.print_exc()
        return False

def plot_solution_diversity(generation_data, output_path):
    """Create a plot showing solution diversity using PCA on network weights,
    colored by fitness.
    
    Args:
        generation_data: List of tuples (generation, list_of_tuples(weight_vector, fitness))
        output_path: Path to save the visualization image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Generate timestamp for the title
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if data is available
        if not generation_data:
            print("No solution diversity data to plot.")
            return False
        
        # Extract data for the latest generation
        latest_gen_data = generation_data[-1]
        generation_num = latest_gen_data[0]
        genome_data_tuples = latest_gen_data[1]
        
        if not genome_data_tuples:
            print(f"No genome data found for generation {generation_num}.")
            return False
            
        # Separate weights and fitness
        weight_vectors = [data[0] for data in genome_data_tuples]
        fitness_values = np.array([data[1] for data in genome_data_tuples])
            
        # Ensure all weight vectors have the same dimension (pad if necessary)
        max_len = max(len(v) for v in weight_vectors)
        padded_vectors = []
        valid_indices = [] # Keep track of indices with valid vectors
        for idx, v in enumerate(weight_vectors):
            if len(v) > 0: # Ensure vector is not empty
                padding = np.zeros(max_len - len(v))
                padded_vectors.append(np.concatenate((v, padding)))
                valid_indices.append(idx)
            else:
                print(f"Warning: Skipping empty weight vector at index {idx} in generation {generation_num}.")
        
        if not padded_vectors:
            print(f"No valid weight vectors after padding for generation {generation_num}.")
            return False
            
        X = np.array(padded_vectors)
        # Filter fitness values to match valid vectors
        filtered_fitness_values = fitness_values[valid_indices]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
        X_pca = pca.fit_transform(X_scaled)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Normalize fitness values for color mapping (0 to 1)
        norm = mcolors.Normalize(vmin=np.min(filtered_fitness_values), vmax=np.max(filtered_fitness_values))
        cmap = cm.viridis # Use a perceptually uniform colormap
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=filtered_fitness_values, cmap=cmap, norm=norm,
                              alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Fitness')
        
        # Add labels and title
        plt.title(f'Solution Diversity (PCA) - Generation {generation_num}\n{timestamp}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return True
        
    except ImportError:
        print("Scikit-learn is required for solution diversity plotting. Please install it (`pip install scikit-learn`).")
        return False
    except (ValueError, TypeError) as e: # Catch potential issues in PCA/scaling/plotting
        print(f"Error processing data for solution diversity plot: {e}")
        return False
    except IOError as e:
        print(f"Error saving solution diversity plot to {output_path}: {e}")
        return False
    except Exception as e:
        import traceback
        print(f"Unexpected error generating solution diversity plot: {e}")
        traceback.print_exc()
        return False 