import torch


def extract_features(obs, max_ships=10, max_planets=8, device='cpu', board_size=100, max_hp=100, max_resource=1000):
    """
    Przetwarza pojedynczą obserwację (dict) na znormalizowany wektor cech i transformuje
    globalne pozycje planet do przestrzeni lokalnej względem pierwszego statku.

    Cechy:
      - Statki: dla maksymalnie max_ships statków pobieramy [pos_x, pos_y, hp] i normalizujemy:
            pos_x, pos_y dzielimy przez board_size, hp dzielimy przez max_hp.
        Jeśli statków jest mniej, uzupełniamy [0, 0, 0].
      - Zasoby: pobieramy pierwszy element z "resources" i normalizujemy przez max_resource.
      - Planety: dla maksymalnie max_planets planet pobieramy:
            [ (planet_x - ref_x)/board_size, (planet_y - ref_y)/board_size, occ_norm ]
        gdzie ref_x, ref_y to globalne współrzędne pierwszego statku. Jeśli planet jest mniej, uzupełniamy [0, 0, -1].
        Dla occupation:
          - jeśli wartość wynosi -1, pozostaje -1,
          - jeśli wynosi 0, pozostaje 0,
          - jeśli wynosi 100, przekształcamy na 1,
          - w przeciwnym razie dzielimy przez 100.

    Łącznie: (max_ships×3) + 1 + (max_planets×3) cech.
    Dla max_ships=10 i max_planets=8 → 10×3 + 1 + 8×3 = 30 + 1 + 24 = 55.
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

    # Ekstrakcja zasobów
    resources_val = obs["resources"][0]
    resources_norm = resources_val / max_resource

    # Ustalamy punkt odniesienia dla transformacji planet – pierwszego statku, jeśli dostępny
    if len(allied_ships) > 0:
        ref_x = allied_ships[0][1]
        ref_y = allied_ships[0][2]
    else:
        ref_x, ref_y = 0, 0

    # Ekstrakcja cech planet – przeliczamy globalne współrzędne na lokalne
    planets = obs.get("planets_occupation", [])
    planet_features = []
    for i in range(max_planets):
        if i < len(planets):
            planet = planets[i]
            # Obliczamy lokalne pozycje względem pierwszego statku
            local_x = (planet[0] - ref_x) / board_size
            local_y = (planet[1] - ref_y) / board_size
            occ = planet[2]
            if occ == -1:
                occ_norm = -1  # niezajęta
            elif occ == 0:
                occ_norm = 0  # zajęta przez gracza 1
            elif occ == 100:
                occ_norm = 1  # zajęta przez gracza 2
            else:
                occ_norm = occ / 100.0  # skalujemy wartości konfliktowe
            planet_features.extend([local_x, local_y, occ_norm])
        else:
            planet_features.extend([0, 0, -1])

    features = ship_features + [resources_norm] + planet_features
    return torch.tensor(features, dtype=torch.float32, device=device)
