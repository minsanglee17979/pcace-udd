#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
set -o noclobber
python -u md-pcace-dcd.py "${1:-../pretrained}" "${2:-40}" > md.out 2>&1

