import pytest
from src.agent import Agent
import torch

@pytest.fixture
def dummy_agent():
    return Agent()

def test_agent_initialization(dummy_agent):
    assert len(list(dummy_agent.parameters())) == 2
    assert dummy_agent.net[0].in_features == 6
    assert dummy_agent.net[2].out_features == 4

def test_forward_pass_shape(dummy_agent):
    obs = {
        'current_pos': [1, 1],
        'goal_pos': [3, 3],
        'walls': [1, 0, 1, 0]
    }
    output = dummy_agent(obs)
    assert output.shape == (4,)
    assert torch.allclose(output.sum(), torch.tensor(1.0))