import torch


def extract_features(obs, max_ships=10, max_planets=8):
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
    resources_val = obs["resources"][0]

    # Ekstrakcja cech planet
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            # Każda planeta to lista: [planet_x, planet_y, occupation_progress]
            planet_features.extend(planets[i])
        else:
            planet_features.extend([0, 0, -1])

    features = ship_features + [resources_val] + planet_features
    return torch.tensor(features, dtype=torch.float32)
