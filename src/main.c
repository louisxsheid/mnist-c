#include <stdio.h>
#include <stdlib.h>
#include "data_io.h"
#include "network.h"
#include "training.h"
#include "utils.h"

int main() {
    // Set seed for reproducibility.
    srand(42);

    printf("📦 Loading MNIST training data...\n");
    MNIST_Dataset train_data = load_mnist_data("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte");
    float **train_normalized = normalize_images(train_data.images, train_data.num_images, train_data.num_rows * train_data.num_cols);

    // Create neural network layers.
    DenseLayer hidden = create_dense_layer(784, 128);
    DenseLayer output = create_dense_layer(128, 10);

    printf("\n🚀 Starting training...\n");
    train_model(&train_data, train_normalized, &hidden, &output, 4);
    save_weights(&hidden, &output, "trained_weights.bin");

    // Free training data.
    for (int i = 0; i < train_data.num_images; i++) {
        free(train_data.images[i]);
        free(train_normalized[i]);
    }
    free(train_data.images);
    free(train_data.labels);
    free(train_normalized);

    printf("\n📦 Loading MNIST test data...\n");
    MNIST_Dataset test_data = load_mnist_data("./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte");
    float **test_normalized = normalize_images(test_data.images, test_data.num_images, test_data.num_rows * test_data.num_cols);

    evaluate_model(&test_data, test_normalized, &hidden, &output);

    // Free test data.
    for (int i = 0; i < test_data.num_images; i++) {
        free(test_data.images[i]);
        free(test_normalized[i]);
    }
    free(test_data.images);
    free(test_data.labels);
    free(test_normalized);

    return 0;
}