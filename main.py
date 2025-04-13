from src.environment import Maze
from src.agent import Agent
from src.evolution import Evolution
from src.visualisation import MazeVisualizer
import numpy as np
import time
import pygame

def simulate_agent(agent, maze, max_steps=100, visualizer=None, agent_idx=None):
    """Simulate a single agent in the maze"""
    if agent.position is None:
        agent.reset(maze.start)
    
    # Run simulation for max_steps or until goal is reached
    for _ in range(max_steps):
        if agent.reached_goal:
            break
            
        # Get observation and determine action
        obs = maze.get_observation(agent.position)
        action_probs = agent(obs).detach().numpy()
        action = np.argmax(action_probs)
        
        # Move agent
        agent.move(action, maze)
        
        # Update visualization if needed
        if visualizer and agent_idx is not None:
            positions = [a.position for a in visualizer.all_agents]
            positions[agent_idx] = agent.position
            visualizer.update(positions, len(positions))
            time.sleep(0.01 / visualizer.simulation_speed)
            
            # Process events to keep UI responsive
            if handle_pygame_events(visualizer):
                return False  # Quit requested
    
    # Calculate and return fitness
    return agent.calculate_fitness(maze)

def simulate_generation(population, maze, visualizer=None):
    """Simulate all agents in the population simultaneously"""
    # Reset all agents
    for agent in population:
        agent.reset(maze.start)
    
    # Store agents in visualizer for updating
    if visualizer:
        visualizer.all_agents = population
    
    # Simulation parameters
    max_steps = maze.size * maze.size * 2  # Plenty of steps
    base_delay = 0.05  # Increased base delay for slower simulation
    
    # Initialize timer
    if visualizer:
        visualizer.max_steps = max_steps
        visualizer.current_step = 0
        visualizer.time_percentage = 100  # Start at 100%
    
    # Run simulation
    step = 0
    while step < max_steps:
        # Always process events to keep UI responsive
        if visualizer:
            # Process events
            running, new_pop_size = visualizer.handle_events()
            if not running:
                return False, new_pop_size  # Quit requested
            
            # Check if we should switch to a different generation
            if visualizer.viewing_generation != visualizer.current_generation:
                return True, new_pop_size  # Signal to switch generations
        
            # If paused, just update display and wait
            if visualizer.paused:
                # Keep the display updated while paused
                positions = [agent.position for agent in population]
                visualizer.update(positions, len(population))
                pygame.event.pump()
                time.sleep(0.05)
                continue
            
            # Update timer when not paused
            visualizer.current_step = step
            visualizer.time_percentage = max(0, round(100 * (1 - step / max_steps)))
        
        # Process all agents
        all_at_goal = True
        
        for i, agent in enumerate(population):
            if agent.reached_goal:
                continue
                
            all_at_goal = False
            
            # Get observation and determine action
            obs = maze.get_observation(agent.position)
            action_probs = agent(obs).detach().numpy()
            action = np.argmax(action_probs)
            
            # Move agent
            agent.move(action, maze)
        
        # Update visualization
        if visualizer:
            positions = [agent.position for agent in population]
            visualizer.update(positions, len(population))
                
            # Control simulation speed - use a non-linear scale for better control
            # This will make slower speeds much slower and faster speeds only slightly faster
            speed_factor = visualizer.simulation_speed
            if speed_factor < 1.0:
                # For values below 1, use a more dramatic slowdown (quadratic scaling)
                delay = base_delay / (speed_factor * speed_factor * 0.1 + 0.1)
            else:
                # For values above 1, scale more gradually
                delay = base_delay / speed_factor
                
            time.sleep(delay)
        
        # Check if all agents have reached the goal
        if all_at_goal:
            # Set timer to 0 since we finished early
            if visualizer:
                visualizer.time_percentage = 0
                visualizer.update(positions, len(population))
            break
        
        # Only increment step if not paused
        step += 1
    
    # Calculate fitness for all agents
    for agent in population:
        agent.calculate_fitness(maze)
    
    return True, visualizer.population_size if visualizer else len(population)

def handle_pygame_events(visualizer):
    """Process pygame events and return True if quit is requested"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

def display_saved_generation(evolution, maze, visualizer, gen_idx):
    """Display a previously saved generation"""
    # Get the saved agents
    saved_agents = evolution.get_generation(gen_idx)
    
    if saved_agents:
        # Set the visualizer to show this generation
        visualizer.viewing_generation = gen_idx
        visualizer.current_generation = gen_idx
        
        # Get positions of all agents
        positions = [agent.position for agent in saved_agents]
        
        # Update visualization with these positions
        visualizer.update(positions, len(saved_agents))
        
        # Update best fitness display
        fitnesses = [agent.fitness for agent in saved_agents]
        if fitnesses:
            visualizer.best_fitness = max(fitnesses)
        
        return True
    return False

def run_simulation(maze_size=10, population_size=30, visualize=True):
    """Run the evolutionary simulation"""
    # Initialize environment
    maze = Maze(size=maze_size)
    
    # Initialize evolution system
    evolution = Evolution(mutation_rate=0.1, elitism_ratio=0.2)
    
    # Initialize population
    population = [Agent() for _ in range(population_size)]
    
    # Setup visualization
    visualizer = None
    if visualize:
        visualizer = MazeVisualizer(maze)
        visualizer.population_size = population_size
        visualizer.current_generation = 0
        visualizer.viewing_generation = 0  # Current generation being displayed
        visualizer.simulation_speed = 0.2
        visualizer.paused = False
        
        # Initial positions for all agents
        for agent in population:
            agent.reset(maze.start)
        visualizer.all_agents = population
    
    # Install matplotlib backend for visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for saving figures
    except ImportError:
        print("Matplotlib not available for neural network visualization")
    
    # Main simulation loop
    running = True
    while running:
        # Handle agent visualization
        if visualizer and visualizer.nn_vis_active:
            # Just process UI events while in visualization mode
            running, new_pop_size = visualizer.handle_events()
            if not running:
                break
            time.sleep(0.05)
            continue
        
        # Check if we should view a saved generation
        if visualizer and visualizer.viewing_generation != visualizer.current_generation:
            if display_saved_generation(evolution, maze, visualizer, visualizer.viewing_generation):
                # Wait for user to return to current generation or quit
                while visualizer.viewing_generation != visualizer.current_generation:
                    running, new_pop_size = visualizer.handle_events()
                    if not running:
                        break
                    time.sleep(0.05)
                continue
            else:
                # If no saved generation, reset view to current
                visualizer.viewing_generation = visualizer.current_generation
        
        # Simulate current generation
        running, new_pop_size = simulate_generation(population, maze, visualizer)
        
        if not running:
            break
            
        # Check for population size change
        if new_pop_size != population_size:
            population_size = new_pop_size
            population = [Agent() for _ in range(population_size)]
            if visualizer:
                visualizer.population_size = population_size
                visualizer.current_generation = 0
                visualizer.viewing_generation = 0
                visualizer.all_agents = population
            continue
        
        # Create next generation
        if not visualizer or not visualizer.paused:
            population, best_fitness = evolution.create_next_gen(population, maze)
            
            if visualizer:
                visualizer.current_generation += 1
                visualizer.viewing_generation = visualizer.current_generation  # Update viewing generation
                visualizer.best_fitness = best_fitness
                visualizer.all_agents = population  # Update reference to current agents
                
                # Print generation stats
                stats = evolution.get_generation_stats()
                print(f"Gen {stats['generation']}: Avg={stats['avg_fitness']:.2f}, Max={stats['max_fitness']:.2f}")
    
    # Clean up
    if visualizer:
        visualizer.quit()

if __name__ == "__main__":
    # Can adjust these parameters
    run_simulation(
        maze_size=10,        # Size of maze (smaller 10x10)
        population_size=20,  # Smaller number of agents
        visualize=True       # Enable visualization
    )
    