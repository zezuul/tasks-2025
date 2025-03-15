import torch.nn as nn
import torch
from config import load_config

config = load_config("simple_agent_1/config.json")


class SimpleAgent(nn.Module):
    def __init__(self, input_dim=config["input_dim"], hidden_dim=config["hidden_dim"]):
        """
        Model przyjmuje wektor cech o wymiarze input_dim.
        Wyjścia:
          - head_action: 2 klasy (0: move, 1: fire)
          - head_direction: 4 klasy (kierunki: 0 - right, 1 - down, 2 - left, 3 - up)
          - head_speed: 3 klasy (dla ruchu: speed 1,2,3)
          - head_construction: 11 klas (0 do 10 statków do konstrukcji)
        """
        super(SimpleAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
        self.head_action = nn.Linear(hidden_dim, 2)
        self.head_direction = nn.Linear(hidden_dim, 4)
        self.head_speed = nn.Linear(hidden_dim, 3)
        self.head_construction = nn.Linear(hidden_dim, 11)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc_shared(x))
        at_logits = self.head_action(x)
        dir_logits = self.head_direction(x)
        spd_logits = self.head_speed(x)
        constr_logits = self.head_construction(x)
        return at_logits, dir_logits, spd_logits, constr_logits
