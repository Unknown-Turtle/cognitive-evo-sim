from src.environment import Maze
from src.agent import Agent
from src.evolution import Evolution
import numpy as np
from src.visualisation import MazeVisualizer
import time
import pygame

def calculate_fitness(agent, maze, visualizer=None):
    position = maze.start
    steps = 0
    for _ in range(50):
        if visualizer:  # Update visualization
            visualizer.update(position)
            time.sleep(0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    visualizer.quit()
                    return 0  # Early exit
                
        obs = maze.get_observation(position)
        action_probs = agent(obs).detach().numpy()
        action = np.argmax(action_probs)  # Deterministic
        
        # Move agent
        new_position = list(position)
        if action == 0 and position[0] > 0: new_position[0] -= 1
        elif action == 1 and position[0] < 4: new_position[0] += 1
        elif action == 2 and position[1] > 0: new_position[1] -= 1
        elif action == 3 and position[1] < 4: new_position[1] += 1
        
        if maze.grid[new_position[0], new_position[1]] == 0:
            position = tuple(new_position)
        
        if position == maze.goal:
            break
            
        steps += 1
    
    return 1/(1 + steps + 0.1*np.linalg.norm(np.array(position) - np.array(maze.goal)))

def train(visualize=False):
    maze = Maze()
    population = [Agent() for _ in range(10)]
    evolution = Evolution()
    
    visualizer = MazeVisualizer(maze) if visualize else None
    
    for generation in range(100):
        fitnesses = [calculate_fitness(agent, maze, visualizer) 
                    if generation % 10 == 0 else  # Only visualize every 10 gens
                    calculate_fitness(agent, maze) 
                    for agent in population]
        
        population = evolution.create_next_gen(population, fitnesses)
        
        if generation % 10 == 0:
            print(f"Gen {generation}: Avg fitness {np.mean(fitnesses):.2f}")

    if visualizer:
        visualizer.quit()

if __name__ == "__main__":
    train(visualize=True)  # Set to False to disable visualization