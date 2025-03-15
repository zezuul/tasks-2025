import pygame
# python run_match.py keyboard_agent.py dummy_agent.py --n_matches=1 --render_mode=human --turn_on_music=True
class Agent:
    def __init__(self):
        # Inicjalizacja Pygame oraz utworzenie okna
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Sterowanie Agentem")

    def get_action(self, obs: dict) -> dict:
        """
        Agent sterowany za pomocą Pygame.
          - Ruch statków: WASD (W - góra, A - lewo, S - dół, D - prawo)
          - Konstrukcja statków: klawisze numeryczne 1-9, a 0 oznacza 10 statków

        Metoda czeka maksymalnie 3 sekundy na naciśnięcie klawisza ruchu.
        Jeśli użytkownik w tym czasie naciśnie także klawisz numeryczny, zapamiętuje wartość konstrukcji.
        """
        ships_actions = []
        allied_ships = obs.get("allied_ships", [])

        direction = None
        construction = 0

        # Mapowanie klawiszy ruchu na kierunki:
        # 0 - prawo, 1 - dół, 2 - lewo, 3 - góra
        key_direction = {
            pygame.K_w: 3,
            pygame.K_a: 2,
            pygame.K_s: 1,
            pygame.K_d: 0
        }

        print("Sterowanie:")
        print("  - Użyj klawiszy WASD do wyboru kierunku ruchu.")
        print("  - Użyj klawiszy numerycznych (1-9, 0 oznacza 10) do ustawienia liczby statków do konstrukcji.")
        print("Czekam na naciśnięcie klawisza ruchu (maks. 3 sekundy)...")
        mode = 0
        start_ticks = pygame.time.get_ticks()
        # Nasłuchuj zdarzeń przez maksymalnie 30000 ms
        while pygame.time.get_ticks() - start_ticks < 30000:
            for event in pygame.event.get():

                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_f]:
                        print("F aktywne")
                        mode = 1
                    else:
                        # Jeśli naciśnięto klawisz ruchu
                        if event.key in key_direction and direction is None:
                            direction = key_direction[event.key]
                            print(f"Wybrany kierunek: {direction} ({'góra' if direction == 3 else 'dół' if direction == 1 else 'lewo' if direction == 2 else 'prawo'})")
                        # Jeśli naciśnięto klawisz numeryczny
                        elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                             pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                            try:
                                num = int(event.unicode)
                                construction = 10 if num == 0 else num
                                print(f"Ustawiono konstrukcję: {construction} statków")
                            except Exception as e:
                                print("Błąd przy odczycie klawisza numerycznego:", e)

            pygame.time.delay(10)
            if direction is not None:
                break

        if direction is None:
            direction = 0
            print("Nie wybrano kierunku. Ustawiam domyślnie: prawo")

        # Przekazujemy akcje jako listy [ship_id, action_type, direction, speed]

        for ship in allied_ships:
            ship_id = ship[0]
            # action_type 0 oznacza ruch, speed ustawiamy na 3
            if mode == 0:
                ships_actions.append([ship_id, 0, direction, 3])
            else:
                print("HELLLLLLEP")
                ships_actions.append([ship_id, 1, direction])


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