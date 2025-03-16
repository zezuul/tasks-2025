import torch
from feature_extraction import extract_features
from simple_agent_1.simple_agent import SimpleAgent


class Agent:
    def __init__(self, player_id):
        self.device = torch.device("cpu")
        # Model przyjmuje wektor cech o wymiarze 55
        self.model = SimpleAgent(input_dim=55, hidden_dim=128)
        self.model.to(self.device)

    def get_action(self, obs: dict) -> dict:
        """
        Dla każdej jednostki z "allied_ships" generuje akcję.
        Dla każdego statku:
          - Wywołuje extract_features z primary_idx ustawionym na indeks statku.
          - Model zwraca predykcje dla akcji, kierunku i prędkości.
          - Jeśli akcja to "move", prędkość = speed_label + 1, w przeciwnym razie 0.
        Globalna etykieta konstrukcji wyznaczana jest dla pierwszego statku.
        Zwraca:
          {"ships_actions": [(ship_id, action_type, direction, speed), ...],
           "construction": construction_value}
        """
        ships_actions = []
        allied_ships = obs.get("allied_ships", [])
        if len(allied_ships) == 0:
            return {"ships_actions": [(0, 0, 0, 1)], "construction": 10}

        for idx, ship in enumerate(allied_ships):
            features = extract_features(obs, max_ships=10, max_planets=8, primary_idx=idx, device=self.device)
            features = features.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                at_logits, dir_logits, spd_logits, constr_logits = self.model(features)
            action_type = torch.argmax(at_logits, dim=1).item()
            direction = torch.argmax(dir_logits, dim=1).item()
            if action_type == 0:
                speed_label = torch.argmax(spd_logits, dim=1).item()
                speed = speed_label + 1
            else:
                speed = 0
            ship_id = ship[0]
            ships_actions.append((ship_id, action_type, direction, speed))

        # Globalna konstrukcja – na podstawie cech pierwszego statku
        features = extract_features(obs, max_ships=10, max_planets=8, primary_idx=0, device=self.device)
        features = features.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            _, _, _, constr_logits = self.model(features)
        construction = torch.argmax(constr_logits, dim=1).item()

        return {"ships_actions": ships_actions, "construction": construction}

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
