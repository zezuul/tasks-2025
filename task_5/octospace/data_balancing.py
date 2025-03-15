import torch


def compute_class_weights(dataset):
    """
    Oblicza wagi klas dla etykiet datasetu.
    Zakładamy, że dataset zwraca:
      - labels: tensor [action_type, direction, speed_label]
      - construction: tensor (int, 0-10)
    Obliczamy wagi osobno dla:
      - action (2 klasy)
      - direction (4 klasy)
      - speed (3 klasy)
      - construction (11 klas)
    """
    action_counts = {}
    direction_counts = {}
    speed_counts = {}
    construction_counts = {}

    for i in range(len(dataset)):
        _, labels, constr = dataset[i]
        a = int(labels[0].item())
        d = int(labels[1].item())
        s = int(labels[2].item())
        c = int(constr.item())

        action_counts[a] = action_counts.get(a, 0) + 1
        direction_counts[d] = direction_counts.get(d, 0) + 1
        speed_counts[s] = speed_counts.get(s, 0) + 1
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

    return (torch.tensor(action_weights, dtype=torch.float32),
            torch.tensor(direction_weights, dtype=torch.float32),
            torch.tensor(speed_weights, dtype=torch.float32),
            torch.tensor(construction_weights, dtype=torch.float32))
