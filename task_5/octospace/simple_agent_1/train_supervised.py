import json
import numpy as np
from pathlib import Path
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset import create_dataloader  # Funkcja zwraca DataLoader z naszym datasetem
from simple_agent_1.simple_agent import SimpleAgent
from data_balancing import compute_class_weights

def train_agent(base_path, num_epochs=10, batch_size=4, learning_rate=1e-3):
    dataloader = create_dataloader(base_path, batch_size=batch_size)

    # Obliczamy wagi klas na podstawie datasetu
    dataset = dataloader.dataset
    action_weights, direction_weights, speed_weights, construction_weights = compute_class_weights(dataset)
    print("Wagi klas:")
    print("  Action weights:", action_weights)
    print("  Direction weights:", direction_weights)
    print("  Speed weights:", speed_weights)
    print("  Construction weights:", construction_weights)

    # Inicjalizacja modelu – input_dim=55, hidden_dim np. 128
    model = SimpleAgent(input_dim=55, hidden_dim=128)
    device = torch.device('cuda')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Funkcje strat z wagami
    loss_action_fn = nn.CrossEntropyLoss(weight=action_weights)
    loss_direction_fn = nn.CrossEntropyLoss(weight=direction_weights)
    loss_speed_fn = nn.CrossEntropyLoss(weight=speed_weights)
    loss_construction_fn = nn.CrossEntropyLoss(weight=construction_weights)

    model.train()
    total_loss_epoch = 0.0
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            # features: [B, max_ships, 55]
            B, M, F = features.shape

            # Forward pass dla akcji statków: spłaszczamy dane do [B*M, 55]
            per_ship_features = features.view(-1, F)
            at_logits, dir_logits, spd_logits, _ = model(per_ship_features)  # ignorujemy output dla konstrukcji tutaj

            # Etykiety: każdy element to tensor [max_ships] – spłaszczamy do [B*M]
            action_labels = labels[0].view(-1)
            direction_labels = labels[1].view(-1)
            speed_labels = labels[2].view(-1)
            construction_labels = labels[3].view(-1)

            loss_action = loss_action_fn(at_logits, action_labels)
            loss_direction = loss_direction_fn(dir_logits, direction_labels)
            mask = (action_labels == 0)
            if mask.sum() > 0:
                loss_speed = loss_speed_fn(spd_logits[mask], speed_labels[mask])
            else:
                loss_speed = 0.0

            # Globalna konstrukcja: wykorzystujemy cechy pierwszego statku dla każdej próbki
            global_features = features[:, 0, :]  # kształt: [B, 55]
            _, _, _, constr_logits = model(global_features)
            loss_construction = loss_construction_fn(constr_logits, construction_labels)

            loss = loss_action + loss_direction + loss_speed + loss_construction
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B
        avg_loss = total_loss / len(dataloader.dataset)
        total_loss_epoch += avg_loss

        # Wyświetlamy informację co 100 epok
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Zapisujemy checkpoint co 10_000 epok
        if (epoch + 1) % 100 == 0:
            checkpoint_dir = Path(base_path) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint zapisany: {checkpoint_path}")

    print("Trening zakończony.")
    return model

if __name__ == '__main__':
    base_path = "sv2"  # Folder z danymi
    # Uwaga: rozmiar batcha 1024*1024 może być zbyt duży, warto dostosować do zasobów GPU.
    trained_model = train_agent(base_path, num_epochs=25_000, batch_size=1024, learning_rate=10e-4)
