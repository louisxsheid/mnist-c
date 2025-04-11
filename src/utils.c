#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

// Read an integer in big-endian format from a file.
int read_big_endian_int(FILE *f) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, f) != 4) {
        perror("fread");
        exit(1);
    }
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Shuffle the dataset and corresponding normalized images.
void shuffle_dataset(MNIST_Dataset *dataset, float **normalized_images) {
    for (int i = dataset->num_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap normalized images.
        float *temp_image = normalized_images[i];
        normalized_images[i] = normalized_images[j];
        normalized_images[j] = temp_image;
        // Swap labels.
        unsigned char temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}

// Save the weights and biases for both hidden and output layers.
void save_weights(DenseLayer *hidden, DenseLayer *output, const char *filename) {
    FILE *fout = fopen(filename, "wb");
    if (!fout) {
        perror("fopen for saving weights");
        exit(1);
    }
    // Save hidden layer weights.
    for (int i = 0; i < hidden->output_size; i++)
        fwrite(hidden->weights[i], sizeof(float), hidden->input_size, fout);
    fwrite(hidden->biases, sizeof(float), hidden->output_size, fout);
    // Save output layer weights.
    for (int i = 0; i < output->output_size; i++)
        fwrite(output->weights[i], sizeof(float), output->input_size, fout);
    fwrite(output->biases, sizeof(float), output->output_size, fout);
    fclose(fout);
    printf("\nSaved weights to %s\n", filename);
}

// Load the weights and biases for both hidden and output layers.
void load_weights(DenseLayer *hidden, DenseLayer *output, const char *filename) {
    FILE *fin = fopen(filename, "rb");
    if (!fin) {
        perror("fopen for loading weights");
        exit(1);
    }
    // Load hidden layer weights.
    for (int i = 0; i < hidden->output_size; i++)
        fread(hidden->weights[i], sizeof(float), hidden->input_size, fin);
    fread(hidden->biases, sizeof(float), hidden->output_size, fin);
    // Load output layer weights.
    for (int i = 0; i < output->output_size; i++)
        fread(output->weights[i], sizeof(float), output->input_size, fin);
    fread(output->biases, sizeof(float), output->output_size, fin);
    fclose(fin);
    printf("Loaded weights from %s\n", filename);
}