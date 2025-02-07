import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, input_size=8, hidden_size=8, output_size=4):  # Change input_size to 8
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs):
        # Flatten all components into a single tensor
        state = torch.cat([
            torch.tensor(obs['current_pos'], dtype=torch.float32),
            torch.tensor(obs['goal_pos'], dtype=torch.float32),
            torch.tensor(obs['walls'], dtype=torch.float32)
        ]).flatten()  # Add .flatten() to ensure 1D tensor
        return self.net(state)
    
    def get_genome(self):
        return [p.data.numpy() for p in self.parameters()]
    
    def set_genome(self, genome):
        for param, weights in zip(self.parameters(), genome):
            param.data = torch.tensor(weights, dtype=torch.float32)  # Add dtype