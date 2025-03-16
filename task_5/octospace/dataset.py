from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from feature_extraction import extract_features

from data import load_data_from_all_matches


class OctoSpaceDataset(Dataset):
    def __init__(self, observations, actions, max_ships=10, max_planets=8):
        """
        Założenia:
          - Każda obserwacja (dict) zawiera: "allied_ships", "resources", "planets_occupation", itd.
            Funkcja extract_features przetwarza obserwację na wektor cech (55 cech) przy użyciu parametru primary_idx.
          - Każda akcja (dict) zawiera:
              "ships_actions" – lista akcji; dla slotu statku spodziewamy się formatu [ship_id, action_type, direction, speed].
              Jeśli dla danego slotu brak akcji, przypisujemy domyślną etykietę [0, 0, 0].
              "construction" – globalna decyzja o budowie statków (int).
        """
        assert len(observations) == len(actions), "Liczba obserwacji musi być równa liczbie akcji."
        self.observations = observations
        self.actions = actions
        self.max_ships = max_ships
        self.max_planets = max_planets
        self.device = torch.device("cuda")
        self.cache = {}

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        obs = self.observations[idx]
        act = self.actions[idx]

        allied_ships = obs.get("allied_ships", [])
        features_list = []
        labels_list = []  # Będzie lista tensorów o wymiarze [3] dla każdego statku

        # Dla każdego slotu statku (od 0 do max_ships-1)
        for i in range(self.max_ships):
            if i < len(allied_ships):
                # Wyciągamy cechy z perspektywy statku o indeksie i
                features = extract_features(obs, max_ships=self.max_ships, max_planets=self.max_planets, primary_idx=i,
                                            device=self.device)
            else:
                features = torch.zeros(55, dtype=torch.float32, device=self.device)
            features_list.append(features)

            # Obsługa akcji: jeśli lista "ships_actions" zawiera akcję dla tego slotu, używamy jej,
            # w przeciwnym razie przypisujemy domyślną etykietę [0, 0, 0] (co odpowiada: move, direction 0, speed_label 0)
            ship_actions = act.get("ships_actions", [])
            if i < len(ship_actions):
                act_i = ship_actions[i]
                # act_i to [ship_id, action_type, direction, speed]
                action_type = act_i[1]
                direction = act_i[2]
                speed_label = (act_i[3] - 1) if action_type == 0 else 0
                label = torch.tensor([action_type, direction, speed_label], dtype=torch.long, device=self.device)
            else:
                label = torch.tensor([0, 0, 0], dtype=torch.long, device=self.device)
            labels_list.append(label)

        # Globalna etykieta konstrukcji
        construction_val = act.get("construction", 0)
        construction_tensor = torch.tensor(construction_val, dtype=torch.long, device=self.device)

        # Teraz zamiast zwracać tensor [max_ships, 3] dla etykiet, rozdzielamy je na trzy 1D tensory:
        action_labels = torch.stack([lbl[0] for lbl in labels_list])  # [max_ships]
        direction_labels = torch.stack([lbl[1] for lbl in labels_list])  # [max_ships]
        speed_labels = torch.stack([lbl[2] for lbl in labels_list])  # [max_ships]

        features_tensor = torch.stack(features_list)  # kształt: [max_ships, 55]
        # Zwracamy tuple z etykietami, które będą gotowe do spłaszczenia w DataLoaderze (batch_size, max_ships)
        data = (features_tensor, (action_labels, direction_labels, speed_labels, construction_tensor))
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