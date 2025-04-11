#include "data_io.h"
#include "utils.h"
#include <stdlib.h>

// Load labels from file.
unsigned char *load_labels(FILE *f, int *num_labels) {
    read_big_endian_int(f); // Skip magic number.
    *num_labels = read_big_endian_int(f);
    unsigned char *labels = malloc(*num_labels);
    if (!labels) {
        perror("malloc for labels");
        exit(1);
    }
    fread(labels, 1, *num_labels, f);
    return labels;
}

// Load images from file.
unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols) {
    read_big_endian_int(f); // Skip magic number.
    *num_images = read_big_endian_int(f);
    *num_rows = read_big_endian_int(f);
    *num_cols = read_big_endian_int(f);
    unsigned char **images = malloc(*num_images * sizeof(unsigned char *));
    if (!images) {
        perror("malloc for images");
        exit(1);
    }
    int image_size = (*num_rows) * (*num_cols);
    for (int i = 0; i < *num_images; i++) {
        images[i] = malloc(image_size);
        if (!images[i]) {
            perror("malloc for image");
            exit(1);
        }
        fread(images[i], 1, image_size, f);
    }
    return images;
}

// Normalize image pixel values to [0, 1].
float **normalize_images(unsigned char **images, int num_images, int size) {
    float **normalized = malloc(num_images * sizeof(float *));
    if (!normalized) {
        perror("malloc for normalized images");
        exit(1);
    }
    for (int i = 0; i < num_images; ++i) {
        normalized[i] = malloc(size * sizeof(float));
        if (!normalized[i]) {
            perror("malloc for normalized image row");
            exit(1);
        }
        for (int j = 0; j < size; ++j) {
            normalized[i][j] = (float)images[i][j] / 255.0f;
        }
    }
    return normalized;
}

// Render a digit image in ASCII.
void render_digit_image_ASCII(unsigned char *image, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int pixel = image[row * cols + col];
            char c = pixel <= 30 ? ' ' : pixel <= 80 ? '.' : pixel <= 160 ? '*' : pixel <= 220 ? 'o' : '#';
            printf("%c", c);
        }
        printf("\n");
    }
}

// Load the entire MNIST dataset.
MNIST_Dataset load_mnist_data(const char *image_path, const char *label_path) {
    MNIST_Dataset dataset;
    FILE *lbl_file = fopen(label_path, "rb");
    if (!lbl_file) { perror("Opening label file"); exit(1); }
    dataset.labels = load_labels(lbl_file, &dataset.num_images);
    fclose(lbl_file);
    
    FILE *img_file = fopen(image_path, "rb");
    if (!img_file) { perror("Opening image file"); exit(1); }
    dataset.images = load_images(img_file, &dataset.num_images, &dataset.num_rows, &dataset.num_cols);
    fclose(img_file);
    
    return dataset;
}