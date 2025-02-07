import pytest
from src.evolution import Evolution
from src.agent import Agent

@pytest.fixture
def evolution():
    return Evolution(mutation_rate=0.1)

def test_mutation(evolution):
    agent = Agent()
    original_genome = agent.get_genome()
    mutated = evolution.mutate(original_genome)
    assert any(not np.array_equal(o, m) for o, m in zip(original_genome, mutated))

def test_crossover(evolution):
    parent1 = [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, 0.0])]
    parent2 = [np.array([[2.0, 2.0], [2.0, 2.0]]), np.array([1.0, 1.0])]
    child = evolution.crossover(parent1, parent2)
    assert any(np.array_equal(layer, parent1[i]) or np.array_equal(layer, parent2[i]) 
              for i, layer in enumerate(child))