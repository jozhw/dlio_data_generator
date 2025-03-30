#!/bin/bash

find_root_dir() {
    local dir="$1"
    while [[ "$dir" != "/" && ! -d "$dir/.git" ]]; do
        dir=$(dirname "$dir")
    done
    echo "$dir"
}

# get project root dir
root_dir=$(find_root_dir "$PWD")

if [ "$PWD" != "$root_dir" ]; then
    echo "You are not in the root project directory. Changing to root directory..."
    cd "$root_dir"
    echo "Changed directory to root project directory: $root_dir"
fi

python3 ./scripts/train.py

Rscript ./src/train.r
