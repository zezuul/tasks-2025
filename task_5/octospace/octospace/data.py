import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def np_encoder(obj):
    """Konwertuje obiekty NumPy na typy kompatybilne z JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def save_training_data_separately(observations, actions, rewards, base_path):
    """
    Zapisuje listę obserwacji, akcje oraz nagrody do trzech osobnych plików JSON.

    Parametry:
      observations: lista dictów z obserwacjami
      actions: lista dictów z akcjami
      rewards: lista nagród (float lub int) dla każdej próbki
      base_path: ścieżka do folderu, w którym mają zostać zapisane pliki
                 (np. "dane_treningowe")

    Funkcja tworzy w podanym folderze trzy pliki:
       - observations.json
       - actions.json
       - rewards.json
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)  # Utwórz folder, jeśli nie istnieje

    obs_filename = base_path / "observations.json"
    actions_filename = base_path / "actions.json"
    rewards_filename = base_path / "rewards.json"

    with open(obs_filename, 'w') as f_obs:
        json.dump(observations, f_obs, indent=2, default=np_encoder)
    with open(actions_filename, 'w') as f_act:
        json.dump(actions, f_act, indent=2, default=np_encoder)
    with open(rewards_filename, 'w') as f_rew:
        json.dump(rewards, f_rew, indent=2, default=np_encoder)

    print(f"Zapisano {len(observations)} obserwacji do pliku: {obs_filename}")
    print(f"Zapisano {len(actions)} akcji do pliku: {actions_filename}")
    print(f"Zapisano {len(rewards)} nagród do pliku: {rewards_filename}")


# Przykładowe dane do zapisu
if __name__ == '__main__':
    observations = []
    actions = []
    # Generujemy przykładowe dane – w realnej symulacji zbierasz te dane w trakcie gry
    for i in range(10):
        obs = {
            "allied_ships": [
                {"ship_id": i, "pos_x": 10 + i, "pos_y": 20 + i, "hp": 100 - i,
                 "firing_cooldown": i % 3, "move_cooldown": i % 2}
            ],
            "resources": 100 + i
        }
        act = {
            "ships_actions": [
                {"ship_id": i, "action_type": 0, "direction": i % 4, "speed": 1}
            ],
            "construction": i % 3
        }
        observations.append(obs)
        actions.append(act)

    save_training_data_separately(observations, actions, "observations.json", "actions.json")


# ---------------------------------------------
# 2. Wczytywanie danych treningowych (osobno)
# ---------------------------------------------
def load_training_data_separately(obs_filename, actions_filename):
    """
    Wczytuje dane treningowe z dwóch plików JSON:
      - obserwacje z obs_filename
      - akcje z actions_filename

    Zwraca:
      observations, actions (dwie listy dictów)
    """
    with open(obs_filename, 'r') as f_obs:
        observations = json.load(f_obs)
    with open(actions_filename, 'r') as f_act:
        actions = json.load(f_act)
    print(f"Wczytano {len(observations)} obserwacji z pliku: {obs_filename}")
    print(f"Wczytano {len(actions)} akcji z pliku: {actions_filename}")
    return observations, actions


# Przykładowe wczytanie danych:
if __name__ == '__main__':
    obs_list, act_list = load_training_data_separately("observations.json", "actions.json")
    print("Przykładowa obserwacja:", obs_list[0])
    print("Przykładowa akcja:", act_list[0])


# ---------------------------------------------
# 3. Dataset w PyTorch – obserwacje i akcje osobno
# ---------------------------------------------
class OctoSpaceDatasetSeparate(Dataset):
    def __init__(self, observations, actions):
        """
        Przyjmuje dwie listy: observations oraz actions.
        Zakładamy, że:
          - Każda obserwacja jest dictem z kluczami: "allied_ships" (lista dictów) oraz "resources".
          - Każda akcja jest dictem z kluczami: "ships_actions" (lista dictów) oraz "construction".
        Dla uproszczenia używamy pierwszego statku i pierwszej akcji z list.
        """
        assert len(observations) == len(actions), "Liczba obserwacji i akcji musi być taka sama."
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]
        # Przygotowanie cech – przykład: bierzemy cechy pierwszego statku oraz zasoby
        ship = obs["allied_ships"][0]
        features = [ship["pos_x"], ship["pos_y"], ship["hp"], obs["resources"]]
        features = torch.tensor(features, dtype=torch.float32)

        # Przygotowanie etykiety – przykład: akcja pierwszego statku (action_type, direction, speed)
        act0 = act["ships_actions"][0]
        # Jeśli w modelu zakładamy indeksację od 0 dla prędkości, można dostosować (tutaj przyjmujemy prędkość bez modyfikacji)
        action_tensor = torch.tensor([act0["action_type"], act0["direction"], act0["speed"]], dtype=torch.long)
        # Liczba statków do konstrukcji jako dodatkowa etykieta
        construction = torch.tensor(act["construction"], dtype=torch.long)

        return features, action_tensor, construction


def create_dataloader_separate(obs_filename, actions_filename, batch_size=2):
    observations, actions = load_training_data_separately(obs_filename, actions_filename)
    dataset = OctoSpaceDatasetSeparate(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Przykładowe wykorzystanie DataLoadera:
if __name__ == '__main__':
    dataloader = create_dataloader_separate("observations.json", "actions.json", batch_size=2)
    for batch in dataloader:
        features, act_tensor, construction = batch
        print("Features:", features)
        print("Akcje:", act_tensor)
        print("Konstrukcja:", construction)
        break
0