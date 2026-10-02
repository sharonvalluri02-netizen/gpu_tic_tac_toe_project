#!/usr/bin/env python3
"""
Two-GPU Tic-Tac-Toe game coordinator.

The coordinator owns the official board and alternates turns.
Each player is a separate GPU process.

On a machine with two GPUs:
    python3 game_runner.py --gpu1 0 --gpu2 1

In the Coursera environment, where only one GPU may be available:
    python3 game_runner.py --gpu1 0 --gpu2 0

The two competitors still execute as independent GPU processes. The
shared-file protocol serializes their turns so they do not require
simultaneous access to a second physical GPU.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def board_to_string(board):
    symbols = {0: ".", 1: "X", 2: "O"}
    return "\n".join(
        " ".join(symbols[x] for x in board[r:r + 3])
        for r in range(0, 9, 3)
    )


def winner(board):
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in lines:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def print_board(board):
    print()
    print(board_to_string(board))
    print()


def atomic_write(path, text):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_move(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = int(f.read().strip())
        return value
    except (FileNotFoundError, ValueError):
        return None


def cleanup(state_dir):
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu1", type=int, default=0)
    parser.add_argument("--gpu2", type=int, default=0)
    parser.add_argument("--dir", default="game_state")
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    state_dir = Path(args.dir)
    cleanup(state_dir)

    board = [0] * 9
    atomic_write(state_dir / "board.txt", " ".join(map(str, board)) + "\n")
    atomic_write(state_dir / "turn.txt", "1\n")

    player_binary = Path("./gpu_player")

    if not player_binary.exists():
        print("ERROR: ./gpu_player does not exist.")
        print("Run 'make' first.")
        return 1

    print("=" * 55)
    print("CUDA TWO-GPU TIC-TAC-TOE")
    print("=" * 55)
    print(f"Player 1 GPU: {args.gpu1}")
    print(f"Player 2 GPU: {args.gpu2}")
    print(f"Shared state:  {state_dir}")
    print("=" * 55)

    p1 = subprocess.Popen([
        str(player_binary),
        "--player", "1",
        "--gpu", str(args.gpu1),
        "--dir", str(state_dir),
    ])

    p2 = subprocess.Popen([
        str(player_binary),
        "--player", "2",
        "--gpu", str(args.gpu2),
        "--dir", str(state_dir),
    ])

    try:
        turn = 1

        for move_number in range(9):
            atomic_write(state_dir / "board.txt",
                         " ".join(map(str, board)) + "\n")
            atomic_write(state_dir / "turn.txt", f"{turn}\n")

            move_file = state_dir / f"move_{turn}.txt"
            start = time.time()

            while True:
                if winner(board) != 0 or 0 not in board:
                    break

                move = read_move(move_file)
                if move is not None:
                    break

                if time.time() - start > 20:
                    raise RuntimeError(
                        f"Player {turn} did not return a move within 20 seconds."
                    )

                time.sleep(0.03)

            move = read_move(move_file)
            if move is None or not (0 <= move <= 8):
                raise RuntimeError(
                    f"Invalid move returned by player {turn}: {move}"
                )

            if board[move] != 0:
                raise RuntimeError(
                    f"Illegal move {move} returned by player {turn}."
                )

            board[move] = turn

            print(f"Turn {move_number + 1}: Player {turn} -> cell {move}")
            print_board(board)

            result = winner(board)
            if result:
                print(f"PLAYER {result} WINS!")
                break

            if 0 not in board:
                print("DRAW!")
                break

            turn = 3 - turn

            # The turn file is updated at the top of the next loop.
            time.sleep(args.delay)

        atomic_write(state_dir / "game_over.txt", "1\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        atomic_write(state_dir / "game_over.txt", "1\n")
    finally:
        try:
            p1.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p1.terminate()

        try:
            p2.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p2.terminate()

        if not (state_dir / "game_over.txt").exists():
            atomic_write(state_dir / "game_over.txt", "1\n")

    print("Game finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
