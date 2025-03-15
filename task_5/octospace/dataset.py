from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from feature_extraction import extract_features

from data import load_data_from_all_matches


class OctoSpaceDataset(Dataset):
    def __init__(self, observations, actions, max_ships=10, max_planets=8):
        """
        Założenia:
         - Każda obserwacja (dict) zawiera klucze: "allied_ships", "resources", "planets_occupation", itd.
           Wektor cech budowany jest przy użyciu funkcji extract_features (wymiar 55).
         - Każda akcja (dict) zawiera:
             "ships_actions" – lista akcji; bierzemy pierwszą akcję (format: [ship_id, action_type, direction, speed])
             oraz "construction" – liczba statków do zbudowania (int, zakres 0-10).
           Jeśli lista "ships_actions" jest pusta, przypisujemy domyślną akcję: [0, 0, 0, 1].
        """
        assert len(observations) == len(actions), "Liczba obserwacji musi być równa liczbie akcji."
        self.observations = observations
        self.actions = actions
        self.max_ships = max_ships
        self.max_planets = max_planets
        self.device = torch.device("cuda")
        # Inicjalizujemy cache – będzie to słownik, w którym kluczem jest indeks, a wartością wynik __getitem__
        self.cache = {}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # Jeśli wynik dla danego indeksu jest już w cache, zwracamy go
        if idx in self.cache:
            return self.cache[idx]

        obs = self.observations[idx]
        act = self.actions[idx]

        # Wyciągamy cechy przy użyciu funkcji extract_features (przekazujemy także device)
        features_tensor = extract_features(obs, max_ships=self.max_ships, max_planets=self.max_planets, device=self.device)

        # Obsługa akcji – lista "ships_actions" może być pusta
        ship_actions = act.get("ships_actions", [])
        if len(ship_actions) > 0:
            act0 = ship_actions[0]
        else:
            act0 = [0, 0, 0, 1]  # Domyślna akcja: [ship_id, action_type, direction, speed]

        # act0: [ship_id, action_type, direction, speed]
        action_type = act0[1]  # 0: move, 1: fire
        direction = act0[2]    # 0-3
        speed_label = (act0[3] - 1) if action_type == 0 else 0
        labels = torch.tensor([action_type, direction, speed_label], dtype=torch.long, device=self.device)

        # Wyciągamy etykietę konstrukcji z akcji (domyślnie 0, jeśli nie podano)
        construction_val = act.get("construction", 0)
        construction_tensor = torch.tensor(construction_val, dtype=torch.long, device=self.device)

        data = (features_tensor, labels, construction_tensor)
        # Zapisujemy wynik w cache
        self.cache[idx] = data
        return data


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