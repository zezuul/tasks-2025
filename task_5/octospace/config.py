import json
from pathlib import Path


def load_config(config_path="config.json"):
    """
    Wczytuje konfigurację z pliku JSON.

    Parametry:
      config_path: ścieżka do pliku konfiguracyjnego (domyślnie "config.json")

    Zwraca:
      config: słownik z ustawieniami
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
    with config_file.open("r") as f:
        config = json.load(f)
    return config
