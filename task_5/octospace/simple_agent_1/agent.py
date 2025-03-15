import torch
from feature_extraction import extract_features
from simple_agent_1.simple_agent import SimpleAgent


class Agent:
    def __init__(self, player_id):
        self.device = torch.device("cpu")
        self.model = SimpleAgent(input_dim=55, hidden_dim=128)
        self.model.to(self.device)

    def get_action(self, obs: dict) -> dict:
        """
        Przetwarza obserwację (dict) na wektor cech, wykonuje predykcję przez wyuczony model
        i zwraca akcję w formacie:
          { "ships_actions": [(ship_id, action_type, direction, speed)], "construction": construction_value }
        """
        features = extract_features(obs, max_ships=10, max_planets=8)
        features = features.unsqueeze(0)  # dodajemy wymiar batch
        features = features.to(self.device)
        self.model.eval()
        with torch.no_grad():
            at_logits, dir_logits, spd_logits, constr_logits = self.model(features)

        action_type = torch.argmax(at_logits, dim=1).item()
        direction = torch.argmax(dir_logits, dim=1).item()
        if action_type == 0:  # move
            speed_label = torch.argmax(spd_logits, dim=1).item()
            speed = speed_label + 1
        else:
            speed = 0

        construction = torch.argmax(constr_logits, dim=1).item()

        if len(obs.get("allied_ships", [])) > 0:
            ship_id = obs["allied_ships"][0][0]
        else:
            ship_id = 0

        return {
            "ships_actions": [(ship_id, action_type, direction, speed)],
            "construction": construction
        }

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
