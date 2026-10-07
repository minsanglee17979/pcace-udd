#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
for name in best_model.pth model_cl.pth model_sr.pth run-train-0.out; do
    if [ -e "$name" ] || [ -L "$name" ]; then
        printf 'Existing output preserved: %s\n' "$name"
        exit 1
    fi
done
set -o noclobber
python -u fit-pcace-cuda.py ../../data/data_Ar.xyz > run-train-0.out 2>&1

