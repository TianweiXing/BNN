CFLAGS=-O3 -lm -std=c99

BinaryCNN: BinaryCNN.c
	gcc $^ -o $@ $(CFLAGS)

.PHONY: clean

clean:
	rm -f BinaryCNN
