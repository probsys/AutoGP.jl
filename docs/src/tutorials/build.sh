#!/usr/bin/env sh

for x in $(ls *.ipynb); do
    python -m jupyter nbconvert --to markdown $x
done

rm -rf .ipynb_checkpoints/
