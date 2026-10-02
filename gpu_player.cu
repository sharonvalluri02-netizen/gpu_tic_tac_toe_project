/*
 * GPU Tic-Tac-Toe competitor
 *
 * Each player is a separate process. The process selects a CUDA device
 * with --gpu and evaluates all legal moves in parallel using a CUDA kernel.
 *
 * The game runner owns the official board. A player only reads board.txt
 * and writes its selected move to move_<player>.txt.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__device__ int winner_device(const int *b) {
    const int lines[8][3] = {
        {0,1,2}, {3,4,5}, {6,7,8},
        {0,3,6}, {1,4,7}, {2,5,8},
        {0,4,8}, {2,4,6}
    };

    for (int i = 0; i < 8; ++i) {
        int a = lines[i][0], c = lines[i][1], d = lines[i][2];
        if (b[a] != 0 && b[a] == b[c] && b[c] == b[d])
            return b[a];
    }
    return 0;
}

__device__ int immediate_win_score(int *board, int move, int me, int opponent) {
    board[move] = me;
    if (winner_device(board) == me)
        return 100000;
    board[move] = opponent;
    if (winner_device(board) == opponent)
        return 90000;
    board[move] = 0;
    return 0;
}

/*
 * One CUDA thread evaluates one board position.
 *
 * Scoring:
 *  - winning move: very high score
 *  - blocking opponent: high score
 *  - center/corners: useful positional score
 *  - deterministic tie-breaking is done by the host
 */
__global__ void evaluate_moves(const int *board, int me, int opponent,
                               int *scores) {
    int move = blockIdx.x * blockDim.x + threadIdx.x;
    if (move >= 9) return;

    if (board[move] != 0) {
        scores[move] = -1000000;
        return;
    }

    int local_board[9];
    for (int i = 0; i < 9; ++i)
        local_board[i] = board[i];

    int score = immediate_win_score(local_board, move, me, opponent);

    if (score == 0) {
        local_board[move] = me;

        // Prefer center.
        if (move == 4) score += 300;

        // Prefer corners.
        if (move == 0 || move == 2 || move == 6 || move == 8)
            score += 200;

        // Prefer edges slightly less than corners.
        if (move == 1 || move == 3 || move == 5 || move == 7)
            score += 100;

        // Reward creating two-in-a-row opportunities.
        const int lines[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8},
            {0,3,6}, {1,4,7}, {2,5,8},
            {0,4,8}, {2,4,6}
        };

        for (int i = 0; i < 8; ++i) {
            int a = lines[i][0], b = lines[i][1], c = lines[i][2];
            int mine = 0, empty = 0;
            if (local_board[a] == me) ++mine;
            if (local_board[b] == me) ++mine;
            if (local_board[c] == me) ++mine;
            if (local_board[a] == 0) ++empty;
            if (local_board[b] == 0) ++empty;
            if (local_board[c] == 0) ++empty;

            if (mine == 2 && empty == 1)
                score += 50;
        }
    }

    scores[move] = score;
}

static bool file_exists(const std::string &path) {
    std::ifstream f(path);
    return f.good();
}

static bool read_board(const std::string &path, std::vector<int> &board) {
    std::ifstream in(path);
    if (!in) return false;

    board.clear();
    int value;
    while (in >> value)
        board.push_back(value);

    return board.size() == 9;
}

static bool write_atomic(const std::string &path, const std::string &content) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) return false;
        out << content;
        out.flush();
        if (!out) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static int parse_arg(int argc, char **argv, const std::string &name, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i])
            return std::atoi(argv[i + 1]);
    }
    return def;
}

static std::string parse_string_arg(int argc, char **argv,
                                    const std::string &name,
                                    const std::string &def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i])
            return argv[i + 1];
    }
    return def;
}

int main(int argc, char **argv) {
    int player = parse_arg(argc, argv, "--player", 1);
    int gpu = parse_arg(argc, argv, "--gpu", 0);
    std::string dir = parse_string_arg(argc, argv, "--dir", "game_state");

    if (player != 1 && player != 2) {
        std::cerr << "Player must be 1 or 2\n";
        return 1;
    }

    std::string board_file = dir + "/board.txt";
    std::string turn_file = dir + "/turn.txt";
    std::string done_file = dir + "/game_over.txt";
    std::string move_file = dir + "/move_" + std::to_string(player) + ".txt";

    CUDA_CHECK(cudaSetDevice(gpu));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));

    std::cout << "Player " << player
              << " using GPU " << gpu
              << ": " << prop.name << std::endl;

    int *d_board = nullptr;
    int *d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_board, 9 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scores, 9 * sizeof(int)));

    while (true) {
        if (file_exists(done_file))
            break;

        std::ifstream turn_in(turn_file);
        int turn = 0;
        if (turn_in)
            turn_in >> turn;

        if (turn != player) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        std::vector<int> board;
        if (!read_board(board_file, board)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // The runner removes our old move before requesting a new one.
        std::remove(move_file.c_str());

        CUDA_CHECK(cudaMemcpy(d_board, board.data(),
                              9 * sizeof(int), cudaMemcpyHostToDevice));

        evaluate_moves<<<1, 9>>>(d_board, player, 3 - player, d_scores);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int scores[9];
        CUDA_CHECK(cudaMemcpy(scores, d_scores,
                              9 * sizeof(int), cudaMemcpyDeviceToHost));

        int best_move = -1;
        int best_score = -10000000;

        // Deterministic tie-breaking: lowest cell number.
        for (int i = 0; i < 9; ++i) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best_move = i;
            }
        }

        if (best_move >= 0 && board[best_move] == 0) {
            std::cout << "Player " << player
                      << " selects cell " << best_move
                      << " with GPU score " << best_score << std::endl;

            write_atomic(move_file, std::to_string(best_move) + "\n");
        }

        // Wait until the runner changes the turn or ends the game.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cudaFree(d_board);
    cudaFree(d_scores);

    std::cout << "Player " << player << " exiting.\n";
    return 0;
}
