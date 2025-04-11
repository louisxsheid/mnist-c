#ifndef TRAINING_H
#define TRAINING_H

#include "data_io.h"
#include "network.h"

// Training and evaluation functions.
float cross_entropy_loss(float *probs, unsigned char label);
float calculate_accuracy(int correct, int total);
void update_output_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float learning_rate);
void backpropagate_to_hidden_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float *hidden_grad);
void update_hidden_layer(DenseLayer *hidden, float *input, float *hidden_grad, float learning_rate);
int train_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output, int epochs);
int evaluate_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output);

#endif /* TRAINING_H */