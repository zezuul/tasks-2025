from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from feature_extraction import extract_features

from data import load_data_from_all_matches


class OctoSpaceDataset(Dataset):
    def __init__(self, observations, actions, max_ships=10, max_planets=8):
        """
        Założenia:
          - Każda obserwacja (dict) zawiera klucze:
                "allied_ships", "resources", "planets_occupation", itd.
            Wektor cech budowany jest przy użyciu funkcji extract_features.
          - Każda akcja (dict) zawiera:
                "ships_actions" – lista akcji; bierzemy pierwszą akcję, która ma format:
                   [ship_id, action_type, direction, speed]
                Jeśli lista jest pusta, przypisujemy domyślną akcję: [0, 0, 0, 1].
        """
        assert len(observations) == len(actions), "Liczba obserwacji musi być równa liczbie akcji."
        self.observations = observations
        self.actions = actions
        self.max_ships = max_ships
        self.max_planets = max_planets

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]

        # Używamy funkcji extract_features do wyciągnięcia cech (tensor o wymiarze 55)
        features_tensor = extract_features(obs, max_ships=self.max_ships, max_planets=self.max_planets)

        # Obsługa akcji – "ships_actions" może być pusta
        ship_actions = act.get("ships_actions", [])
        if len(ship_actions) > 0:
            act0 = ship_actions[0]
        else:
            act0 = [0, 0, 0, 1]  # domyślna akcja: move (action_type 0), direction 0, speed 1 (speed_label = 0)

        # act0: [ship_id, action_type, direction, speed]
        action_type = act0[1]
        direction = act0[2]
        speed = act0[3]
        speed_label = (speed - 1) if action_type == 0 else 0
        labels = torch.tensor([action_type, direction, speed_label], dtype=torch.long)

        return features_tensor, labels


def create_dataloader(base_path, batch_size=4):
    """
    Wczytuje dane z folderu base_path i tworzy DataLoader.
    Oczekiwane pliki w folderze: observations.json, actions.json, reward.json.
    """
    base_path = Path(base_path)
    observations, actions, rewards = load_data_from_all_matches(base_path)
    # reward.json nie jest wykorzystywany w DataLoaderze
    dataset = OctoSpaceDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader