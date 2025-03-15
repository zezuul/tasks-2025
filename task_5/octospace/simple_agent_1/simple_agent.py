import torch
from pathlib import Path
import torch.nn as nn
from feature_extraction import extract_features


# Załóżmy, że wcześniej zdefiniowana jest klasa SimpleAgent
# Model MLP przyjmujący wektor cech o wymiarze 28 i zwracający 3 wyjścia:
# head_action (2 klasy), head_direction (4 klasy) oraz head_speed (3 klasy).
class SimpleAgent(nn.Module):
    def __init__(self, input_dim=55, hidden_dim=32):
        super(SimpleAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
        self.head_action = nn.Linear(hidden_dim, 2)
        self.head_direction = nn.Linear(hidden_dim, 4)
        self.head_speed = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc_shared(x))
        at_logits = self.head_action(x)
        dir_logits = self.head_direction(x)
        spd_logits = self.head_speed(x)
        return at_logits, dir_logits, spd_logits



class Agent:
    def __init__(self, player_id: int):
        """
        Args:
            player_id: Indicates whether agent is player 0 or player 1 in game.
        """
        self.player_id = player_id
        self.device = torch.device("cpu")
        self.model = SimpleAgent(input_dim=55, hidden_dim=32)
        self.model.to(self.device)

    def get_action(self, obs: dict) -> dict:
        """
        Przetwarza obserwację (dict) na wektor cech przy użyciu extract_features,
        wykonuje predykcję przez wyuczony model i zwraca akcję w formacie:
          { "ships_actions": [(ship_id, action_type, direction, speed)], "construction": 0 }
        """
        features = extract_features(obs, max_ships=10, max_planets=8)
        features = features.unsqueeze(0)  # dodajemy wymiar batch
        features = features.to(self.device)
        self.model.eval()
        with torch.no_grad():
            at_logits, dir_logits, spd_logits = self.model(features)

        action_type = torch.argmax(at_logits, dim=1).item()
        direction = torch.argmax(dir_logits, dim=1).item()
        if action_type == 0:  # move
            speed_label = torch.argmax(spd_logits, dim=1).item()
            speed = speed_label + 1
        else:
            speed = 0

        # Wybieramy ship_id z pierwszego statku, jeśli istnieje
        if len(obs.get("allied_ships", [])) > 0:
            ship_id = obs["allied_ships"][0][0]
        else:
            ship_id = 0

        return {"ships_actions": [(ship_id, action_type, direction, speed)], "construction": 0}

    def load(self, abs_path: str):
        from pathlib import Path
        weight_path = Path(abs_path) / "agent_weights.pth"
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        print("Wagi zostały załadowane.")

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)
