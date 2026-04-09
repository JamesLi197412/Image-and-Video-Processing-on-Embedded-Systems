CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -pedantic
BUILD_DIR := build
STATIC_LIB := $(BUILD_DIR)/libimage_ops.a
TEST_BIN := $(BUILD_DIR)/test_image_ops
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
SHARED_EXT := dylib
SHARED_LDFLAGS := -dynamiclib
else
SHARED_EXT := so
SHARED_LDFLAGS := -shared -fPIC
endif

SHARED_LIB := $(BUILD_DIR)/libimage_ops.$(SHARED_EXT)
STATIC_OBJ := $(BUILD_DIR)/image_ops.o
TEST_OBJ := $(BUILD_DIR)/test_image_ops.o

all: $(STATIC_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(STATIC_OBJ): image_ops.c image_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I. -c image_ops.c -o $@

$(STATIC_LIB): $(STATIC_OBJ)
	ar rcs $@ $^

$(SHARED_LIB): image_ops.c image_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHARED_LDFLAGS) -I. image_ops.c -lm -o $@

shared: $(SHARED_LIB)

$(TEST_OBJ): test_image_ops.c image_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I. -c test_image_ops.c -o $@

$(TEST_BIN): $(TEST_OBJ) $(STATIC_LIB)
	$(CC) $(CFLAGS) $(TEST_OBJ) $(STATIC_LIB) -lm -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all shared test clean
