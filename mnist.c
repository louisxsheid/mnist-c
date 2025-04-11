#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Structs for MNIST dataset and neural layers
typedef struct {
    unsigned char **images;
    unsigned char *labels;
    int num_images, num_rows, num_cols;
} MNIST_Dataset;

typedef struct {
    int input_size, output_size;
    float **weights;
    float *biases;
} DenseLayer;

// Function declarations
int read_big_endian_int(FILE *f);
void render_digit_image_ASCII(unsigned char *image, int rows, int cols);
float **normalize_images(unsigned char **images, int num_images, int size);
DenseLayer create_dense_layer(int input_size, int output_size);
void dense_forward(DenseLayer *layer, float *input, float *output);
void relu(float *output, int size);
void softmax(float *logits, int size, float *output_probs);
int predict(float *probs, int size);
float calculate_accuracy(int correct, int total);
float cross_entropy_loss(float *probs, unsigned char label);
void update_output_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float learning_rate);
void backpropagate_to_hidden_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float *hidden_grad);
void update_hidden_layer(DenseLayer *hidden, float *input, float *hidden_grad, float learning_rate);
void shuffle_dataset(MNIST_Dataset *dataset, float **normalized_images);
void save_weights(DenseLayer *hidden, DenseLayer *output, const char *filename);
void load_weights(DenseLayer *hidden, DenseLayer *output, const char *filename);

unsigned char *parse_label_file(FILE *f, int num_labels);
unsigned char **parse_image_file(FILE *f, int num_images, int num_rows, int num_cols);
unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols);
unsigned char *load_labels(FILE *f, int *num_labels);

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

int read_big_endian_int(FILE *f) {
    unsigned char bytes[4];
    fread(bytes, 1, 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

unsigned char *load_labels(FILE *f, int *num_labels) {
    read_big_endian_int(f);
    *num_labels = read_big_endian_int(f);
    unsigned char *labels = malloc(*num_labels);
    fread(labels, 1, *num_labels, f);
    return labels;
}

unsigned char **load_images(FILE *f, int *num_images, int *num_rows, int *num_cols) {
    read_big_endian_int(f);
    *num_images = read_big_endian_int(f);
    *num_rows = read_big_endian_int(f);
    *num_cols = read_big_endian_int(f);
    unsigned char **images = malloc(*num_images * sizeof(unsigned char *));
    for (int i = 0; i < *num_images; i++) {
        images[i] = malloc(*num_rows * *num_cols);
        fread(images[i], 1, *num_rows * *num_cols, f);
    }
    return images;
}

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

DenseLayer create_dense_layer(int input_size, int output_size) {
    DenseLayer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights = malloc(output_size * sizeof(float *));
    for (int i = 0; i < output_size; i++) {
        layer.weights[i] = malloc(input_size * sizeof(float));
        for (int j = 0; j < input_size; j++) {
            layer.weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    layer.biases = calloc(output_size, sizeof(float));
    return layer;
}

void dense_forward(DenseLayer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < layer->input_size; ++j) {
            sum += layer->weights[i][j] * input[j];
        }
        output[i] = sum + layer->biases[i];
    }
}

void relu(float *output, int size) {
    for (int i = 0; i < size; ++i) {
        if (output[i] < 0) output[i] = 0;
    }
}

void softmax(float *logits, int size, float *output_probs) {
    float max = logits[0];
    for (int i = 1; i < size; i++) if (logits[i] > max) max = logits[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) sum_exp += expf(logits[i] - max);
    for (int i = 0; i < size; i++) output_probs[i] = expf(logits[i] - max) / sum_exp;
}

int predict(float *probs, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) if (probs[i] > probs[max_index]) max_index = i;
    return max_index;
}

float calculate_accuracy(int correct, int total) {
    return ((float)correct / total) * 100.0f;
}

float cross_entropy_loss(float *probs, unsigned char label) {
    float epsilon = 1e-7;
    float predicted_prob = probs[label];
    if (predicted_prob < epsilon) predicted_prob = epsilon;
    return -logf(predicted_prob);
}

void shuffle_dataset(MNIST_Dataset *dataset, float **normalized_images) {
    for (int i = dataset->num_images - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        float *temp_image = normalized_images[i];
        normalized_images[i] = normalized_images[j];
        normalized_images[j] = temp_image;
        unsigned char temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}

void update_output_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float learning_rate) {
    for (int j = 0; j < output->output_size; j++) {
        for (int k = 0; k < output->input_size; k++) {
            output->weights[j][k] -= learning_rate * dL_dLogits[j] * hidden_out[k];
        }
        output->biases[j] -= learning_rate * dL_dLogits[j];
    }
}

void backpropagate_to_hidden_layer(DenseLayer *output, float *hidden_out, float *dL_dLogits, float *hidden_grad) {
    for (int j = 0; j < output->input_size; j++) {
        float grad = 0.0f;
        for (int k = 0; k < output->output_size; k++) {
            grad += output->weights[k][j] * dL_dLogits[k];
        }
        hidden_grad[j] = (hidden_out[j] > 0) ? grad : 0.0f;
    }
}

void update_hidden_layer(DenseLayer *hidden, float *input, float *hidden_grad, float learning_rate) {
    for (int j = 0; j < hidden->output_size; j++) {
        for (int k = 0; k < hidden->input_size; k++) {
            hidden->weights[j][k] -= learning_rate * hidden_grad[j] * input[k];
        }
        hidden->biases[j] -= learning_rate * hidden_grad[j];
    }
}

void save_weights(DenseLayer *hidden, DenseLayer *output, const char *filename) {
    FILE *fout = fopen(filename, "wb");
    for (int i = 0; i < hidden->output_size; i++)
        fwrite(hidden->weights[i], sizeof(float), hidden->input_size, fout);
    fwrite(hidden->biases, sizeof(float), hidden->output_size, fout);
    for (int i = 0; i < output->output_size; i++)
        fwrite(output->weights[i], sizeof(float), output->input_size, fout);
    fwrite(output->biases, sizeof(float), output->output_size, fout);
    fclose(fout);
    printf("\n💾 Saved weights to %s\n", filename);
}

int train_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output, int epochs) {
    float learning_rate = 0.001f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_dataset(dataset, normalized_images);
        printf("\nEpoch %d\n", epoch + 1);
        int correct = 0;
        float total_loss = 0.0f;

        for (int i = 0; i < dataset->num_images; ++i) {
            float *input = normalized_images[i];
            float hidden_out[128], hidden_grad[128], logits[10], probs[10];

            dense_forward(hidden, input, hidden_out);
            relu(hidden_out, 128);
            dense_forward(output, hidden_out, logits);
            softmax(logits, 10, probs);

            float expected[10] = {0.0f};
            expected[dataset->labels[i]] = 1.0f;
            float dL_dLogits[10];
            for (int j = 0; j < 10; j++) dL_dLogits[j] = probs[j] - expected[j];

            float loss = cross_entropy_loss(probs, dataset->labels[i]);
            total_loss += loss;

            update_output_layer(output, hidden_out, dL_dLogits, learning_rate);
            backpropagate_to_hidden_layer(output, hidden_out, dL_dLogits, hidden_grad);
            update_hidden_layer(hidden, input, hidden_grad, learning_rate);

            if (predict(probs, 10) == dataset->labels[i]) correct++;
        }

        float accuracy = calculate_accuracy(correct, dataset->num_images);
        printf("✅ Accuracy: %.2f%% | Avg Loss: %.4f\n", accuracy, total_loss / dataset->num_images);
    }
    return 0;
}

int evaluate_model(MNIST_Dataset *dataset, float **normalized_images, DenseLayer *hidden, DenseLayer *output) {
    int correct = 0;
    for (int i = 0; i < dataset->num_images; ++i) {
        float *input = normalized_images[i];
        float hidden_out[128], logits[10], probs[10];

        dense_forward(hidden, input, hidden_out);
        relu(hidden_out, 128);
        dense_forward(output, hidden_out, logits);
        softmax(logits, 10, probs);

        if (predict(probs, 10) == dataset->labels[i]) correct++;
    }
    float accuracy = calculate_accuracy(correct, dataset->num_images);
    printf("\n🔎 Test Accuracy: %.2f%%\n", accuracy);
    return correct;
}

int main() {
    srand(42);
    printf("\U0001F4E6 Loading MNIST training data...\n");
    MNIST_Dataset train_data = load_mnist_data("./mnist/train-images.idx3-ubyte", "./mnist/train-labels.idx1-ubyte");
    float **train_normalized = normalize_images(train_data.images, train_data.num_images, 784);

    DenseLayer hidden = create_dense_layer(784, 128);
    DenseLayer output = create_dense_layer(128, 10);

    printf("\n\U0001F680 Starting training...\n");
    train_model(&train_data, train_normalized, &hidden, &output, 4);
    save_weights(&hidden, &output, "trained_weights.bin");

    for (int i = 0; i < train_data.num_images; i++) {
        free(train_data.images[i]);
        free(train_normalized[i]);
    }
    free(train_data.images);
    free(train_data.labels);
    free(train_normalized);

    printf("\n\U0001F4E6 Loading MNIST test data...\n");
    MNIST_Dataset test_data = load_mnist_data("./mnist/t10k-images.idx3-ubyte", "./mnist/t10k-labels.idx1-ubyte");
    float **test_normalized = normalize_images(test_data.images, test_data.num_images, 784);

    evaluate_model(&test_data, test_normalized, &hidden, &output);

    for (int i = 0; i < test_data.num_images; i++) {
        free(test_data.images[i]);
        free(test_normalized[i]);
    }
    free(test_data.images);
    free(test_data.labels);
    free(test_normalized);

    return 0;
}