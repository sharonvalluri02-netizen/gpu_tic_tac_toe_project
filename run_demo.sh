#!/usr/bin/env bash
set -e

echo "Building CUDA Tic-Tac-Toe..."
make

echo
echo "Running two GPU competitors."
echo "This demo uses GPU 0 for both players so it works on a one-GPU Coursera environment."
echo

python3 game_runner.py --gpu1 0 --gpu2 0
