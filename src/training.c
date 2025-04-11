#include "training.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute the cross-entropy loss.
float cross_entropy_loss(float *probs, unsigned char label) {
    float epsilon = 1e-7f;
    float predicted_prob = probs[label];
    if (predicted_prob < epsilon)
        predicted_prob = epsilon;
    return -logf(predicted_prob);
}

// Calculate accuracy percentage.
float calculate_accuracy(int correct, int total) {
    return ((float)correct / total) * 100.0f;
}

// Update the output layer weights and biases.
void update_output_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float learning_rate) {
    for (int j = 0; j < output->output_size; j++) {
        for (int k = 0; k < output->input_size; k++) {
            output->weights[j][k] -= learning_rate * dL_dLogits[j] * hidden_out[k];
        }
        output->biases[j] -= learning_rate * dL_dLogits[j];
    }
}

// Backpropagate gradients to the hidden layer.
void backpropagate_to_hidden_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float *hidden_grad) {
    for (int j = 0; j < output->input_size; j++) {
        float grad = 0.0f;
        for (int k = 0; k < output->output_size; k++) {
            grad += output->weights[k][j] * dL_dLogits[k];
        }
        hidden_grad[j] = (hidden_out[j] > 0) ? grad : 0.0f;
    }
}

// Update the hidden layer weights and biases.
void update_hidden_layer(DenseLayer *hidden, float *input, float *hidden_grad, float learning_rate) {
    for (int j = 0; j < hidden->output_size; j++) {
        for (int k = 0; k < hidden->input_size; k++) {
            hidden->weights[j][k] -= learning_rate * hidden_grad[j] * input[k];
        }
        hidden->biases[j] -= learning_rate * hidden_grad[j];
    }
}

// Train the model for a number of epochs.
int train_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output, int epochs) {
    float learning_rate = 0.001f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the dataset each epoch.
        shuffle_dataset(dataset, normalized_images);
        printf("\nEpoch %d\n", epoch + 1);
        int correct = 0;
        float total_loss = 0.0f;
        int num_images = dataset->num_images;
        
        for (int i = 0; i < num_images; ++i) {
            float *input = normalized_images[i];
            float hidden_out[128] = {0};
            float hidden_grad[128] = {0};
            float logits[10] = {0};
            float probs[10] = {0};

            // Forward pass: hidden layer.
            dense_forward(hidden, input, hidden_out);
            relu(hidden_out, 128);

            // Forward pass: output layer.
            dense_forward(output, hidden_out, logits);
            softmax(logits, 10, probs);

            // Create one-hot encoding for the expected output.
            float expected[10] = {0};
            expected[dataset->labels[i]] = 1.0f;

            // Compute gradients (dL/dLogits).
            float dL_dLogits[10];
            for (int j = 0; j < 10; j++) {
                dL_dLogits[j] = probs[j] - expected[j];
            }

            float loss = cross_entropy_loss(probs, dataset->labels[i]);
            total_loss += loss;

            // Update output and hidden layers.
            update_output_layer(output, hidden_out, dL_dLogits, learning_rate);
            backpropagate_to_hidden_layer(output, hidden_out, dL_dLogits, hidden_grad);
            update_hidden_layer(hidden, input, hidden_grad, learning_rate);

            if (predict(probs, 10) == dataset->labels[i])
                correct++;
        }

        float accuracy = calculate_accuracy(correct, num_images);
        printf("Accuracy: %.2f%% | Avg Loss: %.4f\n", accuracy, total_loss / num_images);
    }
    return 0;
}

// Evaluate the model performance on the test dataset.
int evaluate_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output) {
    int correct = 0;
    int num_images = dataset->num_images;
    for (int i = 0; i < num_images; ++i) {
        float *input = normalized_images[i];
        float hidden_out[128] = {0};
        float logits[10] = {0};
        float probs[10] = {0};

        dense_forward(hidden, input, hidden_out);
        relu(hidden_out, 128);
        dense_forward(output, hidden_out, logits);
        softmax(logits, 10, probs);

        if (predict(probs, 10) == dataset->labels[i])
            correct++;
    }
    float accuracy = calculate_accuracy(correct, num_images);
    printf("\nTest Accuracy: %.2f%%\n", accuracy);
    return correct;
}