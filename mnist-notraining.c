/* 
🤔 Why 12.83%?
Let’s think probabilistically:
 	•	There are 10 possible classes: digits 0-9
 	•	If your network is making random guesses, you’d expect:
            \frac{1}{10} = 0.10 = 10\%
 	•	But because of the way the weights are initialized (small random values), 
        softmax might slightly favor certain outputs — leading to slightly better-than-random results.
 	•	That small bump — e.g., 12.8% — is just noise, not intelligence.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Struct to hold the entire MNIST dataset in memory
typedef struct {
    unsigned char **images;  // Raw pixel data (0-255)
    unsigned char *labels;   // True digit labels (0-9)
    int num_images;          // Number of images in dataset
    int num_rows;            // Height of each image
    int num_cols;            // Width of each image
} MNIST_Dataset;

// Struct to define a Dense (fully connected) layer
typedef struct {
    int input_size;
    int output_size;
    float **weights;  // weights[output][input]
    float *biases;    // biases[output]
} DenseLayer;

// Function declarations
int read_big_endian_int(FILE *f);
void render_digit_image_ASCII(unsigned char *image, int rows, int cols);
unsigned char *parse_label_file(FILE *f, int num_labels);
unsigned char **parse_image_file(FILE *f, int num_images, int num_rows, int num_cols);
unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols);
unsigned char *load_labels(FILE *f, int *num_labels);
float **normalize_images(unsigned char **images, int num_images, int size);
DenseLayer create_dense_layer(int input_size, int output_size);
void dense_forward(DenseLayer *layer, float *input, float *output);
void relu(float *output, int size);
void softmax(float *logits, int size, float *output_probs);
int predict(float *output_probs, int size);
float calculate_accuracy(int correct, int total);

// ======================= Core Parsing & Utilities =======================

// Read 4-byte int in big-endian format from file
int read_big_endian_int(FILE *f) {
    unsigned char bytes[4];
    fread(bytes, 1, 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Render a 28x28 MNIST digit as ASCII art
void render_digit_image_ASCII(unsigned char *image, int rows, int cols) {
    printf("Rendering digit image as ASCII:\n");
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int pixel = image[row * cols + col];
            char c;
            if      (pixel <= 30)   c = ' ';
            else if (pixel <= 80)   c = '.';
            else if (pixel <= 160)  c = '*';
            else if (pixel <= 220)  c = 'o';
            else                    c = '#';
            printf("%c", c);
        }
        printf("\n");
    }
}

// Parse label file (just raw bytes)
unsigned char *parse_label_file(FILE *f, int num_labels) {
    unsigned char *labels = malloc(num_labels);
    fread(labels, 1, num_labels, f);
    return labels;
}

// Parse image file into image array
unsigned char **parse_image_file(FILE *f, int num_images, int num_rows, int num_cols) {
    unsigned char **images = malloc(num_images * sizeof(unsigned char *));
    for (int i = 0; i < num_images; i++) {
        images[i] = malloc(num_rows * num_cols);
        fread(images[i], 1, num_rows * num_cols, f);
    }
    return images;
}

// Load image file (with header)
unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols) {
    read_big_endian_int(f); // magic number
    *num_images = read_big_endian_int(f);
    *num_rows   = read_big_endian_int(f);
    *num_cols   = read_big_endian_int(f);
    printf("Loaded %d images (%dx%d)\n", *num_images, *num_rows, *num_cols);
    return parse_image_file(f, *num_images, *num_rows, *num_cols);
}

// Load label file (with header)
unsigned char *load_labels(FILE *f, int *num_labels) {
    read_big_endian_int(f); // magic number
    *num_labels = read_big_endian_int(f);
    printf("Loaded %d labels\n", *num_labels);
    return parse_label_file(f, *num_labels);
}

// Normalize image pixels (0-255 → 0.0-1.0)
float **normalize_images(unsigned char **images, int num_images, int size) {
    float **normalized = malloc(num_images * sizeof(float *));
    for (int i = 0; i < num_images; ++i) {
        normalized[i] = malloc(size * sizeof(float));
        for (int j = 0; j < size; ++j) {
            normalized[i][j] = (float)images[i][j] / 255.0f;
        }
    }
    return normalized;
}

// ======================= Neural Network =======================

// Create dense layer with random weights and zero biases
DenseLayer create_dense_layer(int input_size, int output_size) {
    DenseLayer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;

    layer.weights = malloc(output_size * sizeof(float *));
    for (int i = 0; i < output_size; i++) {
        layer.weights[i] = malloc(input_size * sizeof(float));
        for (int j = 0; j < input_size; j++) {
            layer.weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  // [-0.05, 0.05]
        }
    }

    layer.biases = calloc(output_size, sizeof(float));  // initialize biases to 0
    return layer;
}

// Dense layer forward pass: output = weights * input + bias
void dense_forward(DenseLayer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < layer->input_size; ++j) {
            sum += layer->weights[i][j] * input[j];
        }
        output[i] = sum + layer->biases[i];
    }
}

// ReLU activation: max(0, x)
void relu(float *output, int size) {
    for (int i = 0; i < size; ++i) {
        if (output[i] < 0) output[i] = 0;
    }
}

// Softmax: converts logits to class probabilities
void softmax(float *logits, int size, float *output_probs) {
    float max = logits[0];
    for (int i = 1; i < size; i++) if (logits[i] > max) max = logits[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) sum_exp += expf(logits[i] - max);

    for (int i = 0; i < size; i++) {
        output_probs[i] = expf(logits[i] - max) / sum_exp;
    }
}

// Return index of highest probability (predicted digit)
int predict(float *probs, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (probs[i] > probs[max_index]) max_index = i;
    }
    return max_index;
}

// Compute accuracy %
float calculate_accuracy(int correct, int total) {
    return ((float)correct / total) * 100.0f;
}

// ======================= Program Workflow =======================

// Load the full MNIST dataset from files
MNIST_Dataset load_mnist_data() {
    MNIST_Dataset dataset;
    FILE *lbl_file = fopen("./mnist/train-labels.idx1-ubyte", "rb");
    if (!lbl_file) { perror("Opening label file"); exit(1); }
    dataset.labels = load_labels(lbl_file, &dataset.num_images);
    fclose(lbl_file);

    FILE *img_file = fopen("./mnist/train-images.idx3-ubyte", "rb");
    if (!img_file) { perror("Opening image file"); exit(1); }
    dataset.images = load_images(img_file, &dataset.num_images, &dataset.num_rows, &dataset.num_cols);
    fclose(img_file);

    return dataset;
}

// Prediction step for all images
int run_predictions(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output_layer) {
    int correct = 0;

    for (int i = 0; i < dataset->num_images; ++i) {
        float *input = normalized_images[i];
        float hidden_out[128];
        float logits[10];
        float probs[10];

        dense_forward(hidden, input, hidden_out);  // Hidden layer
        relu(hidden_out, 128);                    // Activation
        dense_forward(output_layer, hidden_out, logits); // Output layer
        softmax(logits, 10, probs);               // Convert to class probabilities

        int prediction = predict(probs, 10);       // Get predicted class
        if (prediction == dataset->labels[i]) correct++;
    }

    return correct;
}

// ======================= Entry Point =======================

int main() {
    printf("📦 Loading MNIST data...\n");
    MNIST_Dataset dataset = load_mnist_data();

    printf("🧼 Normalizing images...\n");
    float **normalized_images = normalize_images(dataset.images, dataset.num_images, 784);

    printf("🧠 Initializing neural network...\n");
    DenseLayer hidden = create_dense_layer(784, 128);
    DenseLayer output = create_dense_layer(128, 10);

    printf("🚀 Running predictions...\n");
    int correct = run_predictions(&dataset, normalized_images, &hidden, &output);
    float accuracy = calculate_accuracy(correct, dataset.num_images);
    printf("✅ Accuracy: %.2f%%\n", accuracy);

    // Optionally render sample image
    // render_digit_image_ASCII(dataset.images[12], dataset.num_rows, dataset.num_cols);

    // Free all memory
    for (int i = 0; i < dataset.num_images; i++) {
        free(dataset.images[i]);
        free(normalized_images[i]);
    }
    free(dataset.images);
    free(dataset.labels);
    free(normalized_images);

    return 0;
}

