import torch


def extract_features(obs, max_ships=10, max_planets=8):
    """
    Przetwarza pojedynczą obserwację (dict) na wektor cech.

    Wektor cech budowany jest jako:
      - Statki: dla maksymalnie max_ships statków pobieramy [pos_x, pos_y, hp] (max_ships×3 wartości).
        Jeśli statków jest mniej, uzupełniamy [0, 0, 0].
      - Zasoby: jeśli "resources" jest listą, pobieramy pierwszy element, inaczej wartość.
      - Planety: dla maksymalnie max_planets planet pobieramy [planet_x, planet_y, occupation_progress] (max_planets×3 wartości).
        Jeśli planet jest mniej, uzupełniamy [0, 0, -1].

    Łącznie: (max_ships×3) + 1 + (max_planets×3) cech.
    Dla max_ships=10 i max_planets=8 → 10×3 + 1 + 8×3 = 30 + 1 + 24 = 55.
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
    resources_val = obs["resources"][0]

    # Ekstrakcja cech planet
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            planet_features.extend(planets[i])
        else:
            planet_features.extend([0, 0, -1])

    features = ship_features + [resources_val] + planet_features
    return torch.tensor(features, dtype=torch.float32)
