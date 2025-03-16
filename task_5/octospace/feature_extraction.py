import torch
import math


def get_relative_direction(ship_x, ship_y, target_x, target_y):
    """
    Oblicza dyskretny względny kierunek celu (np. planety) względem pozycji statku.

    Schemat:
      - Jeśli |dx| >= |dy|:
          - 0: right (dx >= 0)
          - 2: left (dx < 0)
      - W przeciwnym razie:
          - 1: down (dy >= 0)
          - 3: up (dy < 0)
    """
    dx = target_x - ship_x
    dy = target_y - ship_y
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0 else 2
    else:
        return 1 if dy >= 0 else 3


def compute_distance(ship_x, ship_y, target_x, target_y):
    """Oblicza euklidesową odległość między dwoma punktami."""
    return math.sqrt((target_x - ship_x) ** 2 + (target_y - ship_y) ** 2)


def extract_features(obs, max_ships=10, max_planets=8, primary_idx=0, device='cpu', board_size=100, max_hp=100,
                     max_resource=1000):
    """
    Przetwarza pojedynczą obserwację (dict) na znormalizowany wektor cech.

    Cechy:
      - Statki: dla maksymalnie max_ships statków pobieramy [pos_x, pos_y, hp] i normalizujemy:
            pos_x, pos_y dzielimy przez board_size, hp przez max_hp.
        Jeśli statków jest mniej, uzupełniamy [0, 0, 0].
      - Zasoby: pobieramy pierwszy element z "resources" (bez weryfikacji typu) i normalizujemy przez max_resource.
      - Planety: dla maksymalnie max_planets planet obliczamy:
            [relative_direction, occ_norm, distance_norm]
        gdzie:
          * relative_direction – dyskretny kierunek (0: right, 1: down, 2: left, 3: up) względem statku o indeksie primary_idx,
          * distance_norm – euklidesowa odległość między referencyjnym statkiem a planetą, podzielona przez board_size,
          * occ_norm – normalizowany postęp zajęcia: -1 pozostaje -1, 0 pozostaje 0, 100 → 1, pozostałe dzielimy przez 100.
        Jeśli planet jest mniej, uzupełniamy [0, -1, 0].

    Łącznie: (max_ships×3) + 1 + (max_planets×3) cech.
    Dla max_ships=10, max_planets=8 → 10×3 + 1 + 8×3 = 30 + 1 + 24 = 55 cech.
    """
    # Ekstrakcja cech statków
    allied_ships = obs.get("allied_ships", [])
    ship_features = []
    for i in range(max_ships):
        if i < len(allied_ships):
            ship = allied_ships[i]
            pos_x_norm = ship[1] / board_size
            pos_y_norm = ship[2] / board_size
            hp_norm = ship[3] / max_hp
            ship_features.extend([pos_x_norm, pos_y_norm, hp_norm])
        else:
            ship_features.extend([0, 0, 0])

    # Ekstrakcja zasobów – zakładamy, że zawsze jest lista
    resources_val = obs["resources"][0]
    resources_norm = resources_val / max_resource

    # Punkt odniesienia dla planet – statek o indeksie primary_idx
    if len(allied_ships) > primary_idx:
        ref_ship = allied_ships[primary_idx]
        ref_x = ref_ship[1]
        ref_y = ref_ship[2]
    else:
        ref_x, ref_y = 0, 0

    # Ekstrakcja cech planet – zamiast lokalnych współrzędnych, obliczamy relative_direction i distance_norm
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            planet = planets[i]  # [planet_x, planet_y, occupation_progress]
            planet_x, planet_y, occ = planet
            rel_dir = float(get_relative_direction(ref_x, ref_y, planet_x, planet_y))
            dist_norm = compute_distance(ref_x, ref_y, planet_x, planet_y) / board_size
            if occ == -1:
                occ_norm = -1.0
            elif occ == 0:
                occ_norm = 0.0
            elif occ == 100:
                occ_norm = 1.0
            else:
                occ_norm = occ / 100.0
            planet_features.extend([rel_dir, occ_norm, dist_norm])
        else:
            planet_features.extend([0, -1, 0])

    features = ship_features + [resources_norm] + planet_features
    return torch.tensor(features, dtype=torch.float32, device=device)
