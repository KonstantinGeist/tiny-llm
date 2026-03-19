NVCC    = nvcc
CC      = gcc
NVFLAGS = -std=c++17 -O3 -Isrc \
          --generate-code arch=compute_75,code=sm_75 \
          --generate-code arch=compute_80,code=sm_80 \
          --generate-code arch=compute_86,code=sm_86 \
          --generate-code arch=compute_89,code=sm_89 \
          --generate-code arch=compute_90,code=sm_90
CFLAGS  = -std=c11 -Wall -Wextra -O2 -Isrc -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm -lpthread -lcudart

# Debug: make DEBUG=1
ifeq ($(DEBUG),1)
NVFLAGS := -std=c++17 -O0 -g -G -Isrc
CFLAGS  := -std=c11 -Wall -Wextra -O0 -g -Isrc -D_POSIX_C_SOURCE=200809L
endif

SRC = src
OBJ = build
BIN = bin

# C sources (compiled with gcc)
C_SRCS = main.c \
         $(SRC)/tokenizer.c \
         $(SRC)/gguf.c \
         $(SRC)/hashmap.c \
         $(SRC)/chat.c \
         $(SRC)/utils.c

# CUDA source (compiled with nvcc)
CU_SRCS = $(SRC)/engine.cu

C_OBJS  = $(OBJ)/main.o \
           $(patsubst $(SRC)/%.c,  $(OBJ)/%.o, $(filter $(SRC)/%, $(C_SRCS)))
CU_OBJS = $(patsubst $(SRC)/%.cu, $(OBJ)/%.o, $(CU_SRCS))

OBJS = $(C_OBJS) $(CU_OBJS)

all: $(BIN)/chat

$(BIN)/chat: $(OBJS) | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ)/main.o: main.c | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ)/%.o: $(SRC)/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ)/%.o: $(SRC)/%.cu | $(OBJ)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

$(OBJ) $(BIN):
	mkdir -p $@

clean:
	find $(OBJ) $(BIN) -mindepth 1 -not -name '.gitignore' -delete

.PHONY: all clean
