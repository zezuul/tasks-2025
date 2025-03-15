import pygame


class Agent:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Sterowanie Agentem - Domyślny ruch / ostrzał")

    def get_action(self, obs: dict) -> dict:
        ships_actions = []
        allied_ships = obs.get("allied_ships", [])

        # Domyślny tryb: ruch (0)
        action_type = 0  # 0 - ruch, 1 - ostrzał
        direction = None
        speed = None
        construction = 0

        # Mapowanie klawiszy kierunkowych (WASD)
        # 0 - prawo, 1 - dół, 2 - lewo, 3 - góra
        key_direction = {
            pygame.K_w: 3,
            pygame.K_a: 2,
            pygame.K_s: 1,
            pygame.K_d: 0
        }

        # Informacja dla użytkownika
        print("Domyślny tryb: Ruch (prędkość = 3).")
        print("Jeśli chcesz wykonać ostrzał, naciśnij F przed wyborem kierunku.")
        print("Wybierz kierunek: WASD (W - góra, A - lewo, S - dół, D - prawo)")

        # Pętla wyboru kierunku – jednocześnie sprawdzamy, czy został naciśnięty F
        direction_selected = False
        while not direction_selected:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        # Zmiana trybu na ostrzał
                        action_type = 1
                        print("Tryb ostrzału wybrany.")
                    elif event.key in key_direction:
                        direction = key_direction[event.key]
                        dir_str = {3: "góra", 1: "dół", 2: "lewo", 0: "prawo"}[direction]
                        print(f"Wybrano kierunek: {direction} ({dir_str})")
                        direction_selected = True
            pygame.time.delay(10)

        # Ustawienie prędkości w zależności od trybu
        if action_type == 0:
            speed = 3  # dla ruchu prędkość jest 3 (domyślna)
            print("Tryb ruchu: prędkość ustawiona na 3.")
        else:
            speed = 0  # dla ostrzału prędkość nie jest istotna
            print("Tryb ostrzału: prędkość ustawiona na 0.")

        # --- Ustawienie konstrukcji statków ---
        print("Wybierz liczbę statków do konstrukcji (klawisze 1-9, 0 oznacza 10) (domyślnie 0):")
        start_ticks = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_ticks < 2000:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                     pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                        try:
                            num = int(event.unicode)
                            construction = 10 if num == 0 else num
                            print(f"Wybrano konstrukcję: {construction} statków")
                        except Exception as e:
                            print("Błąd przy odczycie konstrukcji:", e)
            pygame.time.delay(10)

        # Przypisanie akcji dla każdego statku – zależnie od trybu
        for ship in allied_ships:
            ship_id = ship[0]
            if action_type == 0:
                # Dla ruchu zwracamy krotkę czteroelementową
                ships_actions.append((ship_id, action_type, direction, speed))
            else:
                # Dla ostrzału zwracamy krotkę trzyelementową
                ships_actions.append((ship_id, action_type, direction))

        return {
            "ships_actions": ships_actions,
            "construction": construction
        }

    def load(self, abs_path: str):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass

