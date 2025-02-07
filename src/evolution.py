import random
import numpy as np
from src.agent import Agent

class Evolution:
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate
        
    def select_parents(self, population, fitnesses):
        return random.choices(population, weights=fitnesses, k=2)
    
    def mutate(self, genome):
        return [
            layer + np.random.normal(scale=self.mutation_rate, size=layer.shape).astype(np.float32)  # Add .astype()
            for layer in genome
        ]
    
    def crossover(self, parent1, parent2):
        return [
            (p1 if random.random() < 0.5 else p2)
            for p1, p2 in zip(parent1, parent2)
        ]
    
    def create_next_gen(self, population, fitnesses):
        new_pop = []
        for _ in range(len(population)):
            parents = self.select_parents(population, fitnesses)
            child = self.crossover(parents[0].get_genome(), parents[1].get_genome())
            child = self.mutate(child)
            new_agent = Agent()
            new_agent.set_genome(child)
            new_pop.append(new_agent)
        return new_pop