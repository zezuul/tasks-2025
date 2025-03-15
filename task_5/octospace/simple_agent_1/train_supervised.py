import json
import numpy as np
from pathlib import Path
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset import create_dataloader  # Zakładamy, że funkcja ta zwraca DataLoader z naszym datasetem
from simple_agent_1.simple_agent import SimpleAgent
from data_balancing import compute_class_weights


def train_agent(base_path, num_epochs=10, batch_size=4, learning_rate=1e-3):
    dataloader = create_dataloader(base_path, batch_size=batch_size)

    # Inicjalizujemy dataset, aby obliczyć wagi dla wszystkich głowic
    dataset = dataloader.dataset
    action_weights, direction_weights, speed_weights, construction_weights = compute_class_weights(dataset)
    print("Wagi klas:")
    print("  Action weights:", action_weights)
    print("  Direction weights:", direction_weights)
    print("  Speed weights:", speed_weights)
    print("  Construction weights:", construction_weights)

    model = SimpleAgent(input_dim=55, hidden_dim=128)
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_action_fn = nn.CrossEntropyLoss(weight=action_weights)
    loss_direction_fn = nn.CrossEntropyLoss(weight=direction_weights)
    loss_speed_fn = nn.CrossEntropyLoss(weight=speed_weights)
    loss_construction_fn = nn.CrossEntropyLoss(weight=construction_weights)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels, construction in dataloader:
            optimizer.zero_grad()
            at_logits, dir_logits, spd_logits, constr_logits = model(features)
            a = labels[:, 0]
            loss_action = loss_action_fn(at_logits, a)
            loss_direction = loss_direction_fn(dir_logits, labels[:, 1])
            mask = (labels[:, 0] == 0)
            if mask.sum() > 0:
                loss_speed = loss_speed_fn(spd_logits[mask], labels[mask, 2])
            else:
                loss_speed = 0.0
            loss_construction = loss_construction_fn(constr_logits, construction)

            loss = loss_action + loss_direction + loss_speed + loss_construction
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 1000 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint zapisany: {checkpoint_path}")

    print("Trening zakończony.")
    return model


if __name__ == '__main__':
    base_path = "saves_merge"  # Folder z danymi
    trained_model = train_agent(base_path, num_epochs=10000, batch_size=1024, learning_rate=10e-4)
