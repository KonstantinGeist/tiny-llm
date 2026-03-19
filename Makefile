CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O3 -march=native -mavx2 -mfma -ffast-math \
          -funroll-loops -Isrc -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm -lpthread

# Debug flags (used when DEBUG=1)
ifeq ($(DEBUG),1)
CFLAGS  := -std=c11 -Wall -Wextra -O0 -g -Isrc -D_POSIX_C_SOURCE=200809L
endif

SRC = src
OBJ = build
BIN = bin

SRCS = main.c \
       $(SRC)/engine.c \
       $(SRC)/tokenizer.c \
       $(SRC)/gguf.c \
       $(SRC)/hashmap.c \
       $(SRC)/chat.c

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
