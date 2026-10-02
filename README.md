# CUDA Two-GPU Tic-Tac-Toe

## Assignment

This project implements a simple game in which two GPU-based competitors play
against each other.

The game is **Tic-Tac-Toe**. Each competitor is a separate CUDA process:

- Player 1 uses `X`.
- Player 2 uses `O`.
- Each player runs the same CUDA program with a different player number.
- The players can use different physical GPUs.
- A shared directory is used as the communication/turn mechanism.
- The game runner controls the official board and alternates turns.

The important CUDA part is the `evaluate_moves` kernel. One CUDA thread
evaluates one possible board position, so the nine possible moves are
evaluated in parallel.

---

## Project files

```text
gpu-tic-tac-toe/
├── gpu_player.cu
├── game_runner.py
├── Makefile
├── README.md
└── .gitignore
```

Generated files/directories:

```text
gpu_player
game_state/
```

---

## How the design works

```text
                 ┌──────────────────────┐
                 │    game_runner.py    │
                 │                      │
                 │ owns official board  │
                 │ controls turn order  │
                 └──────────┬───────────┘
                            │
                 shared files / protocol
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │   Player 1      │           │   Player 2      │
     │   gpu_player    │           │   gpu_player    │
     │   CUDA process  │           │   CUDA process  │
     │   GPU 0         │           │   GPU 1         │
     └────────┬────────┘           └────────┬────────┘
              │                             │
              └────────── CUDA ─────────────┘
```

On a system with two GPUs:

```text
Player 1 -> GPU 0
Player 2 -> GPU 1
```

In the Coursera environment, if only one GPU is available, use:

```text
Player 1 -> GPU 0
Player 2 -> GPU 0
```

The players are still independent CUDA processes and the shared turn protocol
prevents both from trying to make a move simultaneously.

---

## CUDA kernel

The kernel is:

```cpp
__global__ void evaluate_moves(...)
```

There are nine possible Tic-Tac-Toe cells.

The kernel launches nine CUDA threads:

```text
Thread 0 -> cell 0
Thread 1 -> cell 1
Thread 2 -> cell 2
...
Thread 8 -> cell 8
```

Each thread evaluates its candidate move.

The score gives priority to:

1. An immediate winning move.
2. Blocking an immediate opponent win.
3. The center.
4. Corners.
5. Edges.
6. Creating two-in-a-row opportunities.

The host program copies the nine scores back from GPU memory and selects the
highest-scoring legal move.

---

## Shared-file turn protocol

The directory `game_state/` contains:

```text
board.txt
turn.txt
move_1.txt
move_2.txt
game_over.txt
```

Example:

```text
board.txt
0 0 1 0 2 0 0 0 0
```

The board uses:

```text
0 = empty
1 = Player 1 / X
2 = Player 2 / O
```

The runner writes:

```text
turn.txt
1
```

Player 1 sees that it is its turn, reads the board, runs its CUDA kernel,
and writes:

```text
move_1.txt
4
```

The runner validates the move, updates the official board, and changes:

```text
turn.txt
2
```

Player 2 then performs the same operation.

This avoids simultaneous writes to the board.

---

## Build

Make sure CUDA/NVCC is available:

```bash
nvcc --version
```

Then:

```bash
make
```

You should get:

```text
gpu_player
```

---

## Run with two physical GPUs

If the machine has GPU 0 and GPU 1:

```bash
python3 game_runner.py --gpu1 0 --gpu2 1
```

or:

```bash
make run
```

---

## Run in Coursera with one GPU

The Coursera environment may expose only one GPU.

Use:

```bash
python3 game_runner.py --gpu1 0 --gpu2 0
```

or:

```bash
make demo
```

This is the recommended command for a single-GPU environment.

---

## Example output

You should see output similar to:

```text
=======================================================
CUDA TWO-GPU TIC-TAC-TOE
=======================================================
Player 1 GPU: 0
Player 2 GPU: 0
Shared state:  game_state
=======================================================
Player 1 using GPU 0: NVIDIA ...
Player 2 using GPU 0: NVIDIA ...

Player 1 selects cell 4 with GPU score 300
Turn 1: Player 1 -> cell 4

. . .
. X .
. . .

Player 2 selects cell 0 with GPU score 200
Turn 2: Player 2 -> cell 0

O . .
. X .
. . .

...
```

The exact moves depend on the board and scoring.

---

## Why two processes?

The assignment asks for two GPU competitors.

Using two independent processes makes the design easy to demonstrate:

```text
gpu_player --player 1 --gpu 0
gpu_player --player 2 --gpu 1
```

Each process independently:

1. Selects a CUDA device.
2. Copies the board to GPU memory.
3. Launches a CUDA kernel.
4. Evaluates candidate moves.
5. Copies the scores back.
6. Produces one move.

The runner acts as the referee.

---

## Why use files?

The assignment specifically mentions that lock files or another mechanism can
be used to allow GPU programs to take turns.

This implementation uses a simple turn-file protocol:

```text
turn = 1
   |
   v
Player 1 calculates
   |
   v
move_1.txt
   |
   v
Runner updates board
   |
   v
turn = 2
   |
   v
Player 2 calculates
   |
   v
move_2.txt
```

Files are written atomically using a temporary file followed by `os.replace()`.
This prevents the other process from observing a partially written file.

---

## Presentation/demo outline

A short presentation can use these sections.

### 1. Problem

"Two GPU-based competitors play a simple game of Tic-Tac-Toe."

### 2. Architecture

Show:

```text
             Game Runner
             /         \
            /           \
       Player 1       Player 2
        GPU 0           GPU 1
```

Explain that the runner is the referee and the players are independent
CUDA processes.

### 3. CUDA kernel

Show:

```cpp
evaluate_moves<<<1, 9>>>(...)
```

Explain that nine CUDA threads evaluate nine possible board cells.

### 4. GPU strategy

Explain the scoring:

```text
Winning move       -> highest priority
Blocking move      -> next priority
Center             -> preferred
Corner             -> preferred
Edge               -> lower priority
```

### 5. Communication

Show:

```text
board.txt
turn.txt
move_1.txt
move_2.txt
```

Explain that only the player whose number appears in `turn.txt` is allowed
to generate a move.

### 6. Demonstration

Run:

```bash
make
make demo
```

Then explain that `make demo` uses GPU 0 for both processes because the
Coursera environment can expose only one physical GPU.

For a two-GPU machine:

```bash
python3 game_runner.py --gpu1 0 --gpu2 1
```

### 7. Limitations and future work

Possible extensions:

- Minimax search on the GPU.
- Monte Carlo Tree Search.
- Larger games such as Connect Four.
- Different strategies for each competitor.
- More CUDA threads and deeper game-tree evaluation.
- CUDA streams for overlapping data transfers and computation.

---

## Troubleshooting

### `nvcc: command not found`

CUDA is not available in the current environment. Use the CUDA-enabled
Coursera environment/container.

### `invalid device ordinal`

The selected GPU does not exist.

Check:

```bash
nvidia-smi
```

Then use valid GPU IDs.

For a single GPU:

```bash
python3 game_runner.py --gpu1 0 --gpu2 0
```

### `./gpu_player does not exist`

Run:

```bash
make
```

### Player waits forever

Remove old game state and restart:

```bash
rm -rf game_state
make demo
```

### Stop a stuck game

Press:

```text
Ctrl+C
```

Then:

```bash
rm -rf game_state
```

---

## Clean build

```bash
make clean
make
```

---

## GitHub upload

After creating a GitHub repository:

```bash
git init
git add .
git commit -m "CUDA two-GPU Tic-Tac-Toe"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Do not commit generated files such as `gpu_player` or `game_state/`.
The included `.gitignore` handles them.
