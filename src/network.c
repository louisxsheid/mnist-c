#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Create a DenseLayer with random weights and zero biases.
DenseLayer create_dense_layer(int input_size, int output_size) {
    DenseLayer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights = malloc(output_size * sizeof(float *));
    if (!layer.weights) {
        perror("malloc for weights");
        exit(1);
    }
    for (int i = 0; i < output_size; i++) {
        layer.weights[i] = malloc(input_size * sizeof(float));
        if (!layer.weights[i]) {
            perror("malloc for weights row");
            exit(1);
        }
        for (int j = 0; j < input_size; j++) {
            layer.weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    layer.biases = calloc(output_size, sizeof(float));
    if (!layer.biases) {
        perror("calloc for biases");
        exit(1);
    }
    return layer;
}

// Compute the forward pass through a dense layer.
void dense_forward(DenseLayer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < layer->input_size; ++j) {
            sum += layer->weights[i][j] * input[j];
        }
        output[i] = sum + layer->biases[i];
    }
}

// Apply the ReLU activation function.
void relu(float *output, int size) {
    for (int i = 0; i < size; ++i) {
        if (output[i] < 0)
            output[i] = 0;
    }
}

// Compute the softmax of the input logits.
void softmax(float *logits, int size, float *output_probs) {
    float max = logits[0];
    for (int i = 1; i < size; i++)
        if (logits[i] > max)
            max = logits[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(logits[i] - max);
    }
    for (int i = 0; i < size; i++) {
        output_probs[i] = expf(logits[i] - max) / sum_exp;
    }
}

// Return the index of the maximum probability.
int predict(float *probs, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++)
        if (probs[i] > probs[max_index])
            max_index = i;
    return max_index;
}