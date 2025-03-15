# Plik: feature_extraction.py
import torch
import math


def get_relative_direction(ship_x, ship_y, target_x, target_y):
    """
    Oblicza dyskretny względny kierunek celu (planety lub wrogiego statku)
    względem pozycji statku.

    Schemat:
      - Jeśli |dx| >= |dy|, kierunek poziomy:
          - 0: right (dx >= 0)
          - 2: left (dx < 0)
      - W przeciwnym razie kierunek pionowy:
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


def extract_features(obs, max_ships=10, max_planets=8, max_enemy_ships=5):
    """
    Przetwarza pojedynczą obserwację (dict) na wektor cech.

    Składowe wektora cech:
      - Allied ships: dla maksymalnie max_ships statków pobieramy [pos_x, pos_y, hp] (max_ships×3 wartości).
        Jeśli statków jest mniej, uzupełniamy [0, 0, 0].
      - Resources: 1 wartość – jeśli "resources" jest listą, bierzemy pierwszy element.
      - Planets: dla maksymalnie max_planets planet obliczamy:
            [relative_direction, occupation_progress, distance]
          relative_direction – dyskretny kierunek (0: right, 1: down, 2: left, 3: up) względem pierwszego statku.
          Jeśli planet jest mniej, uzupełniamy [0, -1, 0].
      - Enemy ships: dla maksymalnie max_enemy_ships wrogich statków obliczamy:
            [distance, relative_direction]
          Jeśli wrogich statków jest mniej, uzupełniamy [0, 0].

    Łącznie cechy = (max_ships*3) + 1 + (max_planets*3) + (max_enemy_ships*2).
    Dla domyślnych parametrów: 10×3 + 1 + 8×3 + 5×2 = 30 + 1 + 24 + 10 = 65.
    """
    # Ekstrakcja cech statków
    allied_ships = obs.get("allied_ships", [])
    ship_features = []
    for i in range(max_ships):
        if i < len(allied_ships):
            ship = allied_ships[i]
            # Zakładamy, że statek to lista: [ship_id, pos_x, pos_y, hp, firing_cooldown, move_cooldown]
            ship_features.extend([ship[1], ship[2], ship[3]])
        else:
            ship_features.extend([0, 0, 0])

    # Ekstrakcja zasobów
    resources_val = obs["resources"][0] if isinstance(obs["resources"], list) else obs["resources"]

    # Ustalamy pozycję głównego statku (pierwszy statek)
    if len(allied_ships) > 0:
        primary_ship = allied_ships[0]
        ship_x = primary_ship[1]
        ship_y = primary_ship[2]
    else:
        ship_x, ship_y = 0, 0

    # Ekstrakcja cech planet: zamiast absolutnych pozycji, kodujemy relative_direction oraz odległość
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            # Każda planeta to lista: [planet_x, planet_y, occupation_progress]
            planet_info = planets[i]
            planet_x, planet_y, occ = planet_info
            rel_dir = get_relative_direction(ship_x, ship_y, planet_x, planet_y)
            dist = compute_distance(ship_x, ship_y, planet_x, planet_y)
            planet_features.extend([rel_dir, occ, dist])
        else:
            planet_features.extend([0, -1, 0])

    # Ekstrakcja cech wrogich statków: kodujemy [distance, relative_direction] względem głównego statku
    enemy_ships = obs.get("enemy_ships", [])
    enemy_features = []
    for i in range(max_enemy_ships):
        if i < len(enemy_ships):
            enemy = enemy_ships[i]
            # Zakładamy, że format enemy ship: [ship_id, pos_x, pos_y, hp, firing_cooldown, move_cooldown]
            enemy_x = enemy[1]
            enemy_y = enemy[2]
            dist = compute_distance(ship_x, ship_y, enemy_x, enemy_y)
            rel_dir = get_relative_direction(ship_x, ship_y, enemy_x, enemy_y)
            enemy_features.extend([dist, rel_dir])
        else:
            enemy_features.extend([0, 0])

    features = ship_features + [resources_val] + planet_features + enemy_features
    return torch.tensor(features, dtype=torch.float32)
