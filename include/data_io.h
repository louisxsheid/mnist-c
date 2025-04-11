#ifndef DATA_IO_H
#define DATA_IO_H

#include <stdio.h>

// Structure for MNIST dataset.
typedef struct {
    unsigned char **images;
    unsigned char *labels;
    int num_images, num_rows, num_cols;
} MNIST_Dataset;

// Function prototypes for data I/O.
MNIST_Dataset load_mnist_data(const char *image_path, const char *label_path);
unsigned char *load_labels(FILE *f, int *num_labels);
unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols);
float **normalize_images(unsigned char **images, int num_images, int size);
void render_digit_image_ASCII(unsigned char *image, int rows, int cols);

#endif /* DATA_IO_H */