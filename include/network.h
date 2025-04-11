#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

// Structure for a fully connected (dense) layer.
typedef struct {
    int input_size, output_size;
    float **weights;
    float *biases;
} DenseLayer;

// Neural network layer functions.
DenseLayer create_dense_layer(int input_size, int output_size);
void dense_forward(DenseLayer *layer, float *input, float *output);
void relu(float *output, int size);
void softmax(float *logits, int size, float *output_probs);
int predict(float *probs, int size);

#endif /* NETWORK_H */