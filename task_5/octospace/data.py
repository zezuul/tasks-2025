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


def load_training_data_separately(base_path):
    """
    Wczytuje dane treningowe z folderu base_path.
    Oczekiwane pliki:
       - observations.json
       - actions.json
       - reward.json  (nagroda jako pojedynczy float – nie jest używana w Dataset)
    Zwraca:
       observations, actions, reward
    """
    base_path = Path(base_path)
    with open(base_path / "observations.json", 'r') as f_obs:
        observations = json.load(f_obs)
    with open(base_path / "actions.json", 'r') as f_act:
        actions = json.load(f_act)
    with open(base_path / "reward.json", 'r') as f_rew:
        reward = json.load(f_rew)
    return observations, actions, reward


def load_data_from_single_match(match_folder: Path):
    """
    Wczytuje dane treningowe z jednego folderu rozgrywki (match_folder).
    Wewnątrz folderu znajdują się podfoldery rund (np. 0, 1, 2, ... 10),
    a w każdym z nich:
      - observations.json
      - actions.json
      - rewards.json
    Zwraca:
      match_observations, match_actions, match_rewards
      (listy zsumowanych danych ze wszystkich rund w danej rozgrywce)
    """
    match_observations = []
    match_actions = []
    match_rewards = []

    # Iterujemy po podfolderach rund
    for round_folder in match_folder.iterdir():
        if round_folder.is_dir():
            obs_file = round_folder / "observations.json"
            act_file = round_folder / "actions.json"
            rew_file = round_folder / "rewards.json"

            # Sprawdzamy, czy wszystkie pliki istnieją
            if obs_file.exists() and act_file.exists() and rew_file.exists():
                try:
                    with open(obs_file, 'r') as f_obs:
                        observations = json.load(f_obs)
                    with open(act_file, 'r') as f_act:
                        actions = json.load(f_act)
                    with open(rew_file, 'r') as f_rew:
                        reward = json.load(f_rew)

                    match_observations.append(observations)
                    match_actions.append(actions)
                    match_rewards.append(reward)  # Zakładamy, że w pliku jest pojedynczy float lub cokolwiek w JSON

                except Exception as e:
                    print(f"  Błąd przy wczytywaniu rundy {round_folder.name} w rozgrywce {match_folder.name}: {e}")
            else:
                print(f"  Brak plików .json w rundzie {round_folder.name} (pomijam).")

    return match_observations, match_actions, match_rewards


def load_data_from_all_matches(main_folder: str):
    """
    Wczytuje dane z folderu `main_folder` (np. "pierwszekrokibebika"),
    w którym znajdują się podfoldery reprezentujące osobne rozgrywki
    (np. 2025-03-15_16-41-48, 2025-03-15_16-50-00, itp.).

    Każdy z tych podfolderów zawiera kolejne rundy (0..10 itd.) z plikami
    observations.json, actions.json i rewards.json.

    Zwraca:
      all_observations, all_actions, all_rewards
      (zsumowane dane ze wszystkich rozgrywek i wszystkich rund)
    """
    base_path = Path(main_folder)
    all_observations = []
    all_actions = []
    all_rewards = []

    # Iterujemy po podfolderach – każda nazwa to np. 2025-03-15_16-41-48
    for match_subfolder in base_path.iterdir():
        if match_subfolder.is_dir():
            print(f"[Rozgrywka {match_subfolder.name}]")
            try:
                obs, acts, rews = load_data_from_single_match(match_subfolder)
                all_observations.extend(obs)
                all_actions.extend(acts)
                all_rewards.extend(rews)
            except Exception as e:
                print(f"  Błąd przy wczytywaniu rozgrywki {match_subfolder.name}: {e}")

    print(f"\nŁącznie wczytano {len(all_observations)} obserwacji, "
          f"{len(all_actions)} akcji z {len(all_rewards)} rund we wszystkich rozgrywkach.")
    return all_observations, all_actions, all_rewards

