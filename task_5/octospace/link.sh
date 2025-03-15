#!/bin/bash

# Pobieranie argumentów
src="$1"
dst="$2"

# Sprawdzanie, czy argumenty zostały podane
if [ -z "$src" ] || [ -z "$dst" ]; then
    echo "Użycie: $0 <ścieżka_do_src> <ścieżka_do_dst>"
    exit 1
fi

# Iteracja przez foldery w src
for dir in "$src"/*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")

        mkdir -p "$dst/$base_name"
        mount --bind "$dir" "$dst/$base_name"
    fi
done

echo "Linki utworzono pomyślnie, zachowując oryginalne nazwy!"
