import torch


def compute_class_weights(dataset, device="cuda"):
    """
    Oblicza wagi klas dla etykiet z datasetu.

    Zakładamy, że dataset zwraca tuple:
      - features_tensor: [max_ships, feature_dim]
      - labels: tuple (action_labels, direction_labels, speed_labels) o kształcie [max_ships] każdy
      - construction: tensor (scalar, int, 0-10)

    Obliczamy wagi osobno dla:
      - action (2 klasy)
      - direction (4 klasy)
      - speed (3 klasy)
      - construction (11 klas)

    Wagi liczymy jako:
        waga = total_count / (num_classes * count_danej_klasy)

    Zwracamy tensory wag umieszczone na urządzeniu device.
    """
    action_counts = {}
    direction_counts = {}
    speed_counts = {}
    construction_counts = {}

    for i in range(len(dataset)):
        _, labels = dataset[i]
        # labels to tuple (action_labels, direction_labels, speed_labels), każdy [max_ships]
        action_labels, direction_labels, speed_labels, constr = labels

        # Iterujemy po wszystkich statkach (slotach)
        for j in range(action_labels.size(0)):
            a = int(action_labels[j].item())
            d = int(direction_labels[j].item())
            s = int(speed_labels[j].item())
            action_counts[a] = action_counts.get(a, 0) + 1
            direction_counts[d] = direction_counts.get(d, 0) + 1
            speed_counts[s] = speed_counts.get(s, 0) + 1

        c = int(constr.item())
        construction_counts[c] = construction_counts.get(c, 0) + 1

    total_actions = sum(action_counts.values())
    total_directions = sum(direction_counts.values())
    total_speeds = sum(speed_counts.values())
    total_construction = sum(construction_counts.values())

    num_action_classes = 2
    num_direction_classes = 4
    num_speed_classes = 3
    num_construction_classes = 11

    action_weights = []
    for i in range(num_action_classes):
        count = action_counts.get(i, 1)
        weight = total_actions / (num_action_classes * count)
        action_weights.append(weight)

    direction_weights = []
    for i in range(num_direction_classes):
        count = direction_counts.get(i, 1)
        weight = total_directions / (num_direction_classes * count)
        direction_weights.append(weight)

    speed_weights = []
    for i in range(num_speed_classes):
        count = speed_counts.get(i, 1)
        weight = total_speeds / (num_speed_classes * count)
        speed_weights.append(weight)

    construction_weights = []
    for i in range(num_construction_classes):
        count = construction_counts.get(i, 1)
        weight = total_construction / (num_construction_classes * count)
        construction_weights.append(weight)

    return (torch.tensor(action_weights, dtype=torch.float32, device=device),
            torch.tensor(direction_weights, dtype=torch.float32, device=device),
            torch.tensor(speed_weights, dtype=torch.float32, device=device),
            torch.tensor(construction_weights, dtype=torch.float32, device=device))
