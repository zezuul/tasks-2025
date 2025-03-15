import torch


def get_relative_direction(ship_x, ship_y, planet_x, planet_y):
    """
    Oblicza dyskretny względny kierunek planety względem pozycji statku.

    Zasada:
      - Jeśli |dx| >= |dy|, kierunek poziomy:
          - 0: prawa (dx >= 0)
          - 2: lewa (dx < 0)
      - W przeciwnym razie kierunek pionowy:
          - 1: dół (dy >= 0)
          - 3: góra (dy < 0)
    """
    dx = planet_x - ship_x
    dy = planet_y - ship_y
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0 else 2
    else:
        return 1 if dy >= 0 else 3


def extract_features(obs, max_ships=10, max_planets=8, device='cpu'):
    """
    Przetwarza pojedynczą obserwację (dict) na wektor cech.

    Argumenty:
      obs: dict zawierający klucze "allied_ships", "resources", "planets_occupation", itd.
      max_ships: maksymalna liczba statków do uwzględnienia (domyślnie 10)
      max_planets: maksymalna liczba planet do uwzględnienia (domyślnie 8)

    Wektor cech budowany jest jako:
      - Statki: dla każdego z maksymalnie 10 statków pobieramy [pos_x, pos_y, hp] (10×3 = 30 wartości).
        Jeśli statków jest mniej, uzupełniamy [0, 0, 0].
      - Zasoby: jeśli "resources" jest listą, pobieramy pierwszy element, w przeciwnym razie wartość bezpośrednio (1 wartość).
      - Planety: dla maksymalnie 8 planet pobieramy [planet_x, planet_y, occupation_progress] (8×3 = 24 wartości).
        Jeśli planet jest mniej, uzupełniamy [0, 0, -1].

    Łącznie: 30 + 1 + 24 = 55 cech.
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

    # Ekstrakcja zasobów – jeśli "resources" jest listą, bierzemy pierwszy element
    resources_val = obs["resources"][0] if isinstance(obs["resources"], list) else obs["resources"]

    # Ustalamy referencyjną pozycję statku – używamy pierwszego statku, jeśli istnieje
    if len(allied_ships) > 0:
        primary_ship = allied_ships[0]
        ship_x = primary_ship[1]
        ship_y = primary_ship[2]
    else:
        ship_x, ship_y = 0, 0

    # Ekstrakcja cech planet: zamiast absolutnych pozycji, obliczamy relative_direction
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            # Każda planeta to lista: [planet_x, planet_y, occupation_progress]
            planet_features.extend(planets[i])
        else:
            planet_features.extend([0, -1])

    features = ship_features + [resources_val] + planet_features
    return torch.tensor(features, dtype=torch.float32, device=device)
