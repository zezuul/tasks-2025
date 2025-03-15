import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# ============================
# 1. Model decyzyjny – MLP
# ============================
class DecisionModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=9):
        """
        Model przyjmuje wektor cech o wymiarze 6:
        [pos_x, pos_y, hp, firing_cooldown, move_cooldown, resources]
        A następnie generuje 9 wyjść, podzielonych na:
          - 2 wartości dla wyboru akcji (0: ruch, 1: strzał),
          - 4 wartości dla kierunku (0: prawo, 1: dół, 2: lewo, 3: góra),
          - 3 wartości dla prędkości (dla ruchu, odpowiadające prędkości 1-3).
        """
        super(DecisionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # wyjście: 2 + 4 + 3 = 9

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # Podział na trzy segmenty:
        action_type_logits = out[:, :2]
        direction_logits = out[:, 2:6]
        speed_logits = out[:, 6:9]
        return action_type_logits, direction_logits, speed_logits


# ============================
# 2. Klasa Agent
# ============================
class Agent:
    def __init__(self):
        self.model = DecisionModel()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_action(self, obs: dict) -> dict:
        """
        Dla każdego statku spróbuj wygenerować akcję na podstawie wektora cech:
        [pos_x, pos_y, hp, firing_cooldown, move_cooldown, resources]
        Wyjścia modelu dzielimy na:
         - akcja: 0 - ruch, 1 - strzał;
         - kierunek: 0 - prawo, 1 - dół, 2 - lewo, 3 - góra;
         - prędkość: dla ruchu (dla strzału prędkość = 0).
        """
        actions = []
        allied_ships = obs.get("allied_ships", [])
        resources = obs.get("resources", 0)
        for ship in allied_ships:
            # ship: (ship_id, pos_x, pos_y, hp, firing_cooldown, move_cooldown)
            ship_id, pos_x, pos_y, hp, fire_cd, move_cd = ship
            feature = torch.tensor([pos_x, pos_y, hp, fire_cd, move_cd, resources],
                                   dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                at_logits, dir_logits, spd_logits = self.model(feature)
                # Obliczenie rozkładów prawdopodobieństwa:
                at_prob = torch.softmax(at_logits, dim=1)
                dir_prob = torch.softmax(dir_logits, dim=1)
                spd_prob = torch.softmax(spd_logits, dim=1)
                # Losowanie akcji:
                action_type = torch.multinomial(at_prob, num_samples=1).item()  # 0 - move, 1 - fire
                direction = torch.multinomial(dir_prob, num_samples=1).item()  # 0-3
                if action_type == 0:
                    # dla ruchu prędkość losujemy z zakresu 1-3 (indeksujemy spd_prob, dodajemy 1)
                    speed = torch.multinomial(spd_prob, num_samples=1).item() + 1
                else:
                    speed = 0
            actions.append({
                "ship_id": ship_id,
                "action_type": action_type,
                "direction": direction,
                "speed": speed
            })
        # Prosty warunek budowy nowych statków – buduj, gdy zasoby są powyżej progu
        construction = 1 if resources > 50 else 0
        return {
            "ships_actions": actions,
            "construction": construction
        }

    def load(self, abs_path: str):
        """Ładowanie wag modelu z pliku."""
        self.model.load_state_dict(torch.load(abs_path, map_location=self.device))

    def eval(self):
        """Przełączenie modelu w tryb ewaluacji."""
        self.model.eval()

    def to(self, device):
        """Przeniesienie modelu na GPU lub inne urządzenie."""
        self.device = device
        self.model.to(device)


# ============================
# 3. Pętla ucząca na podstawie danych z pliku
# ============================
def training_loop_from_file(agent, data_file="training_data.csv", epochs=10, batch_size=32):
    """
    Plik CSV powinien zawierać kolumny:
      pos_x, pos_y, hp, firing_cooldown, move_cooldown, resources,
      action_type (0 lub 1), direction (0-3), speed (1-3)
    """
    df = pd.read_csv(data_file)
    X = df[["pos_x", "pos_y", "hp", "firing_cooldown", "move_cooldown", "resources"]].values.astype(np.float32)
    y_action = df["action_type"].values.astype(np.int64)
    y_direction = df["direction"].values.astype(np.int64)
    y_speed = df["speed"].values.astype(np.int64)  # dla ruchu; przy strzale można ignorować

    # Podział na zbiór treningowy i walidacyjny
    X_train, X_val, y_action_train, y_action_val, y_direction_train, y_direction_val, y_speed_train, y_speed_val = train_test_split(
        X, y_action, y_direction, y_speed, test_size=0.2, random_state=42)

    device = agent.device
    model = agent.model
    optimizer = agent.optimizer
    criterion = nn.CrossEntropyLoss()

    train_size = X_train.shape[0]
    for epoch in range(epochs):
        permutation = np.random.permutation(train_size)
        epoch_loss = 0.0
        for i in range(0, train_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X = torch.tensor(X_train[indices], device=device)
            batch_y_action = torch.tensor(y_action_train[indices], device=device)
            batch_y_direction = torch.tensor(y_direction_train[indices], device=device)
            # Przesunięcie etykiety prędkości do zakresu [0,2]
            batch_y_speed = torch.tensor(y_speed_train[indices] - 1, device=device)

            optimizer.zero_grad()
            at_logits, dir_logits, spd_logits = model(batch_X)
            loss_action = criterion(at_logits, batch_y_action)
            loss_direction = criterion(dir_logits, batch_y_direction)
            loss_speed = criterion(spd_logits, batch_y_speed)
            loss = loss_action + loss_direction + loss_speed
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / train_size:.4f}")
    print("Trening na danych z pliku zakończony.")


# ============================
# 4. Pętla ucząca z wykorzystaniem rywalizacji agentów (self-play)
# ============================
def simulate_game(agent1, agent2):
    """
    Dummy symulacja gry między agentami.
    W rzeczywistości należałoby podpiąć symulację środowiska.
    Tutaj zwracamy losowe nagrody dla demonstracji.
    """
    reward_agent1 = np.random.rand()
    reward_agent2 = np.random.rand()
    return reward_agent1, reward_agent2


def training_loop_competition(agent, num_games=100, gamma=0.99):
    """
    Prosta pętla treningowa wykorzystująca policy gradient
    (na bardzo uproszczonym przykładzie) do aktualizacji modelu w wyniku gry.
    """
    optimizer = agent.optimizer
    device = agent.device
    model = agent.model
    model.train()

    for game in range(num_games):
        # Tworzymy kopię drugiego agenta (można też użyć self-play)
        agent1 = agent
        agent2 = Agent()
        agent2.model.load_state_dict(model.state_dict())
        agent2.to(device)

        # Symulacja gry między agentami
        reward1, reward2 = simulate_game(agent1, agent2)

        # Dla demonstracji – używamy dummy input i zakładamy, że akcja została podjęta,
        # obliczamy log-probabilitę tej akcji i modyfikujemy wagę na podstawie nagrody.
        dummy_input = torch.tensor([[50, 50, 100, 0, 0, 100]], device=device, dtype=torch.float32)
        at_logits, dir_logits, spd_logits = model(dummy_input)
        log_prob_at = torch.log_softmax(at_logits, dim=1)[0, 0]  # zakładamy, że wybrano akcję 0 (ruch)
        log_prob_dir = torch.log_softmax(dir_logits, dim=1)[0, 0]  # kierunek 0
        log_prob_spd = torch.log_softmax(spd_logits, dim=1)[0, 0]  # prędkość 1 (indeks 0)
        log_prob = log_prob_at + log_prob_dir + log_prob_spd

        # Obliczenie straty – minimalizujemy -reward * log_prob
        loss = -reward1 * log_prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Gra {game + 1}/{num_games}: Nagroda: {reward1:.3f}, Strata: {loss.item():.4f}")
    print("Trening przy rywalizacji zakończony.")


# ============================
# 5. Przykładowe wywołanie
# ============================
if __name__ == "__main__":
    agent = Agent()

    # Przykładowa obserwacja – można ją wykorzystać do testowania get_action
    obs_example = {
        "allied_ships": [
            (1, 10, 20, 80, 0, 0),
            (2, 15, 25, 90, 2, 1)
        ],
        "resources": 100,
        "game_map": None,  # tutaj nie korzystamy z mapy
        "enemy_ships": [],
        "planets_occupation": []
    }
    action = agent.get_action(obs_example)
    print("Wygenerowana akcja:", action)


