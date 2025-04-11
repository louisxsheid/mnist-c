#ifndef UTILS_H
#define UTILS_H

#include "data_io.h"
#include "network.h"
#include <stdio.h>

// Utility functions.
int read_big_endian_int(FILE *f);
void shuffle_dataset(MNIST_Dataset *dataset, float **normalized_images);
void save_weights(DenseLayer *hidden, DenseLayer *output, const char *filename);
void load_weights(DenseLayer *hidden, DenseLayer *output, const char *filename);

#endif /* UTILS_H */