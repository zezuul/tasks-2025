import json
import numpy as np
from pathlib import Path
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from dataset import *  # Zakładamy, że create_dataloader jest zdefiniowany w tym module
from simple_agent import SimpleAgent


def train_agent(base_path, num_epochs=10, batch_size=4, learning_rate=1e-3):
    dataloader = create_dataloader(base_path, batch_size=batch_size)

    model = SimpleAgent(input_dim=55, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    # Utwórz folder na checkpointy, jeśli nie istnieje
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            at_logits, dir_logits, spd_logits = model(features)
            loss_action = criterion(at_logits, labels[:, 0])
            loss_direction = criterion(dir_logits, labels[:, 1])
            mask = (labels[:, 0] == 0)
            if mask.sum() > 0:
                loss_speed = criterion(spd_logits[mask], labels[mask, 2])
            else:
                loss_speed = 0.0
            loss = loss_action + loss_direction + loss_speed
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Zapisz checkpoint co 100 epok
        if (epoch + 1) % 1000 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint zapisany: {checkpoint_path}")

    print("Trening zakończony.")
    return model


###########################################
# Przykładowe wywołanie treningu
###########################################

if __name__ == '__main__':
    base_path = "pierwszekrokibejbika"  # Folder z danymi
    trained_model = train_agent(base_path, num_epochs=10000, batch_size=128, learning_rate=1e-3)
