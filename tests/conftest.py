import pytest
import torch
from src.agent import Agent
from src.evolution import Evolution
from src.environment import Maze

# --------------------------
# Core Fixtures (Reusable across tests)
# --------------------------

@pytest.fixture(scope="session")
def maze():
    """Provides a reusable maze instance for all tests."""
    return Maze()

@pytest.fixture
def dummy_agent():
    """Creates a fresh agent instance for each test."""
    return Agent()

@pytest.fixture
def evolution():
    """Creates an evolution instance with default mutation rate."""
    return Evolution(mutation_rate=0.1)

# --------------------------
# Advanced Fixtures
# --------------------------

@pytest.fixture
def trained_population(maze):
    """Returns a pre-evolved population of agents."""
    population = [Agent() for _ in range(10)]
    evolution = Evolution()
    for _ in range(5):  # Simulate 5 generations
        fitnesses = [calculate_fitness(agent, maze) for agent in population]
        population = evolution.create_next_gen(population, fitnesses)
    return population

# --------------------------
# Configuration & Hooks
# --------------------------

def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption("--seed", action="store", default=42, help="Set random seed")

@pytest.fixture(autouse=True)
def set_seed(request):
    """Automatically set random seeds for reproducibility."""
    seed = request.config.getoption("--seed")
    torch.manual_seed(seed)
    # If using numpy elsewhere:
    # import numpy as np
    # np.random.seed(seed)
    

