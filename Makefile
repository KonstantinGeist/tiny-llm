CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -Isrc -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm -lpthread

# Debug flags (used when DEBUG=1)
ifeq ($(DEBUG),1)
CFLAGS  := -std=c11 -Wall -Wextra -O0 -g -Isrc -D_POSIX_C_SOURCE=200809L
endif

SRC = src
OBJ = build
BIN = bin

# Core sources (always compiled)
SRCS = main.c \
       $(SRC)/engine.c \
       $(SRC)/tokenizer.c \
       $(SRC)/gguf.c \
       $(SRC)/hashmap.c \
       $(SRC)/chat.c \
       $(SRC)/math_common.c \
       $(SRC)/utils.c

# Math backend selection (pick at most one):
#   make           → math_cpu.c      (pure C, no deps)
#   make BLAS=1    → math_openblas.c (requires -lopenblas)
ifeq ($(BLAS),1)
SRCS    += $(SRC)/math_openblas.c
CFLAGS  += -DUSE_OPENBLAS
LDFLAGS += -lopenblas
else
SRCS    += $(SRC)/math_cpu.c
endif

OBJS = $(OBJ)/main.o $(filter-out main.c,$(SRCS:$(SRC)/%.c=$(OBJ)/%.o))

all: $(BIN)/chat

$(BIN)/chat: $(OBJS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ)/main.o: main.c | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ)/%.o: $(SRC)/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ) $(BIN):
	mkdir -p $@

clean:
	find $(OBJ) $(BIN) -mindepth 1 -not -name '.gitignore' -delete

.PHONY: all clean
