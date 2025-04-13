import random
import numpy as np
import copy
from src.agent import Agent

class Evolution:
    def __init__(self, mutation_rate=0.1, elitism_ratio=0.1):
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.generation_history = []  # Store information about each generation
        self.saved_generations = {}  # Store agents from each generation
        self.current_generation = 0
        
    def select_parents(self, population, fitnesses):
        """Select parents using tournament selection"""
        # Scale the fitnesses to ensure they're positive
        scaled_fitnesses = np.array(fitnesses) - min(fitnesses) + 1e-6
        
        # Tournament selection
        tournament_size = max(2, len(population) // 5)
        parents = []
        
        for _ in range(2):  # Select two parents
            # Randomly select tournament_size individuals
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [scaled_fitnesses[i] for i in tournament_indices]
            
            # Select the winner
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
            
        return parents
    
    def mutate(self, genome, strength=1.0):
        """Mutate the genome with adaptive mutation strength"""
        return [
            layer + np.random.normal(scale=self.mutation_rate * strength, size=layer.shape).astype(np.float32)
            for layer in genome
        ]
    
    def crossover(self, parent1, parent2):
        """Perform uniform crossover between parents"""
        child = []
        parent1_genome = parent1.get_genome()
        parent2_genome = parent2.get_genome()
        
        for p1_layer, p2_layer in zip(parent1_genome, parent2_genome):
            # Create mask for uniform crossover
            mask = np.random.randint(0, 2, size=p1_layer.shape).astype(np.bool_)
            child_layer = np.copy(p1_layer)
            child_layer[mask] = p2_layer[mask]
            child.append(child_layer)
            
        return child
    
    def create_next_gen(self, population, maze):
        """Create a new generation of agents based on their performance"""
        # First calculate and update all fitnesses
        fitnesses = []
        for agent in population:
            if agent.position is None:  # Skip agents without position
                continue
            fitness = agent.calculate_fitness(maze)
            fitnesses.append(fitness)
        
        if not fitnesses:  # If no valid fitnesses, return the same population
            return population, 0.0
        
        # Store generation data
        gen_data = {
            'generation': self.current_generation,
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses)
        }
        self.generation_history.append(gen_data)
        
        # Save current generation's agents
        self.save_generation(population)
        
        # Apply elitism - keep best performers
        elitism_count = max(1, int(self.elitism_ratio * len(population)))
        elite_indices = np.argsort(fitnesses)[-elitism_count:]
        elite_agents = [population[i] for i in elite_indices]
        
        # Create new population
        new_pop = []
        
        # First add the elite agents (unchanged)
        for agent in elite_agents:
            new_agent = Agent()
            new_agent.set_genome(agent.get_genome())
            new_pop.append(new_agent)
        
        # Then fill the rest with offspring
        while len(new_pop) < len(population):
            parents = self.select_parents(population, fitnesses)
            child_genome = self.crossover(parents[0], parents[1])
            
            # Adaptive mutation - higher for poor performers
            mutation_strength = 1.0
            if parents[0].fitness < gen_data['avg_fitness'] and parents[1].fitness < gen_data['avg_fitness']:
                mutation_strength = 1.5  # Increase mutation for weaker parents
                
            child_genome = self.mutate(child_genome, mutation_strength)
            new_agent = Agent()
            new_agent.set_genome(child_genome)
            new_pop.append(new_agent)
        
        self.current_generation += 1
        return new_pop, np.max(fitnesses)
    
    def save_generation(self, population):
        """Save the current generation's agents for later viewing"""
        # Make deep copies of all agents to preserve their exact state
        saved_agents = []
        for agent in population:
            # Create a new agent with the same properties
            saved_agent = Agent()
            saved_agent.set_genome(agent.get_genome())
            
            # Copy all relevant properties
            saved_agent.fitness = agent.fitness
            saved_agent.position = agent.position
            saved_agent.best_position = agent.best_position
            saved_agent.steps_taken = agent.steps_taken
            saved_agent.reached_goal = agent.reached_goal
            
            saved_agents.append(saved_agent)
            
        # Store in dictionary with generation number as key
        self.saved_generations[self.current_generation] = saved_agents
    
    def get_generation(self, generation_idx):
        """Get the agents from a specific generation"""
        if generation_idx in self.saved_generations:
            return self.saved_generations[generation_idx]
        return None
    
    def get_generation_stats(self, generation_idx=None):
        """Get statistics for a specific generation or the current one"""
        if generation_idx is None:
            generation_idx = self.current_generation - 1
            
        if 0 <= generation_idx < len(self.generation_history):
            return self.generation_history[generation_idx]
            
        return {'generation': 0, 'avg_fitness': 0, 'max_fitness': 0, 'min_fitness': 0}