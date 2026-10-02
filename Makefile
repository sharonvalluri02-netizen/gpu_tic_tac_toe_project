NVCC ?= nvcc
CUDA_ARCH ?=

NVCC_FLAGS = -std=c++17 -O2 -Xcompiler -Wall

ifneq ($(strip $(CUDA_ARCH)),)
NVCC_FLAGS += -arch=$(CUDA_ARCH)
endif

.PHONY: all build clean run demo

all: build

build: gpu_player

gpu_player: gpu_player.cu
	$(NVCC) $(NVCC_FLAGS) gpu_player.cu -o gpu_player

run: build
	python3 game_runner.py --gpu1 0 --gpu2 1

demo: build
	python3 game_runner.py --gpu1 0 --gpu2 0

clean:
	rm -f gpu_player
	rm -rf game_state
