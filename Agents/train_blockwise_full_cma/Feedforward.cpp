// Feedforward.cpp
// A lightweight fully-connected feedforward neural network implementation in pure C++.
// Supports arbitrary depth and width, tanh activations throughout,
// and flat parameter vectors for use with black-box optimisers (e.g. CMA-ES).
//
// Developed with assistance from:
//   Claude  (Anthropic)  — https://www.anthropic.com
//   ChatGPT (OpenAI)     — https://openai.com
//   Gemini  (Google)     — https://deepmind.google

#include <vector>
#include <cmath>
#include <stdexcept>


// =============================================================================
// Neuron
// =============================================================================

class neuron {

    private:

        // Computes the dot product of two equal-length vectors.
        double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
            if (v1.size() != v2.size()) {
                throw std::invalid_argument("Vectors must be of the same length for dot product.");
            }
            double sum = 0.0;
            for (size_t i = 0; i < v1.size(); i++) {
                sum += v1[i] * v2[i];
            }
            return sum;
        }

        // Tanh activation — maps any real value to (-1, 1).
        double tanh_activation(double x) {
            return std::tanh(x);
        }

    public:

        double              bias;     // scalar bias term
        std::vector<double> weights;  // one weight per input

        // Constructor — stores the neuron's bias and weight vector.
        neuron(double bias, std::vector<double> weights) {
            this->bias    = bias;
            this->weights = weights;
        }

        // Computes: tanh( dot(inputs, weights) + bias )
        double forward(const std::vector<double>& inputs) {
            double z = dot_product(inputs, this->weights) + this->bias;
            return tanh_activation(z);
        }
};


// =============================================================================
// Layer
// =============================================================================

class layer {

    public:

        std::vector<neuron> neurons;  // a vector of neurons constitutes this layer

        // Constructs a layer of `size` neurons, each expecting `input_layer_dim` inputs.
        // All weights and biases initialised to 0.0 — CMA-ES sets them before any forward pass.
        layer(int size, int input_layer_dim) {
            for (int i = 0; i < size; i++) {
                double              bias = 0.0;
                std::vector<double> weights(input_layer_dim, 0.0);
                this->neurons.push_back(neuron(bias, weights));
            }
        }

        // Runs each neuron's forward pass and returns the layer's output vector.
        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> outputs;
            for (neuron& n : this->neurons) {
                outputs.push_back(n.forward(inputs));
            }
            return outputs;
        }
};


// =============================================================================
// NeuralNetwork
// =============================================================================

class neural_network {

    public:

        std::vector<layer> layers;  // hidden layers followed by output layer
        int                block_size;  // number of neurons per block for block-diagonal CMA-ES

        // Constructs a fully-connected network:
        //   input_size          → dimension of the observation vector
        //   hidden_layer_sizes  → e.g. {64, 64} for two hidden layers of 64 neurons each
        //   output_size         → dimension of the action vector
        //   block_size          → number of neurons per block when partitioning each layer;
        //                         except possibly the last block of each layer, which holds
        //                         the remaining neurons if the layer size is not exactly divisible by block_size
        //
        // All weights are initialised to 0.0 — use set_param() to load CMA-ES solutions.
        neural_network(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size, int block_size) {
            this->block_size = block_size;
            int prev_size    = input_size;

            // Hidden layers.
            for (int size : hidden_layer_sizes) {
                layers.push_back(layer(size, prev_size));
                prev_size = size;
            }

            // Output layer — tanh keeps actions within (-1, 1),
            // matching the normalised action spaces of MuJoCo and Box2D environments.
            layers.push_back(layer(output_size, prev_size));
        }

        // Partitions parameters into blocks, layer by layer.
        // For each layer, neurons are grouped into blocks of block_size.
        // Each block contains the bias + weights of up to block_size neurons from that layer only —
        // neurons from different layers are never mixed in the same block.
        // The last block of each layer holds the remainder neurons if the layer size is not
        // divisible by block_size.
        // // Returns a vector of vectors where each inner vector is a block of parameters.
        std::vector<std::vector<double>> get_param() {
            std::vector<std::vector<double>> blocks;

            for (layer& l : this->layers) {
                size_t num_neurons = l.neurons.size();
                size_t i           = 0;

                // Step through this layer's neurons in chunks of block_size.
                while (i < num_neurons) {
                    size_t end = std::min(i + (size_t)this->block_size, num_neurons);  // last block may be smaller

                    // Collect bias + weights for neurons i .. end-1 into one block.
                    std::vector<double> block;
                    for (size_t j = i; j < end; j++) {
                        block.push_back(l.neurons[j].bias);
                        for (double w : l.neurons[j].weights) {
                            block.push_back(w);
                        }
                    }
                    blocks.push_back(block);
                    i = end;
                }
            }

            return blocks;
        }

        // Takes a vector of blocks, flattens them back into a single parameter vector,
        // then loads the parameters into the network.
        // Order must match get_param(): bias then weights, neuron by neuron, layer by layer.
        void set_param(const std::vector<std::vector<double>>& blocks) {

            // Step 1: flatten blocks back into a single vector.
            std::vector<double> flat;
            for (const std::vector<double>& block : blocks) {
                for (double val : block) {
                    flat.push_back(val);
                }
            }

            // Step 2: load flat vector into network.
            size_t index = 0;
            for (layer& l : this->layers) {
                for (neuron& n : l.neurons) {
                    if (index >= flat.size()) {
                        throw std::out_of_range("Not enough parameters provided to set_param.");
                    }
                    n.bias = flat[index++];            // post-increment: reads index, then increments
                    for (size_t w = 0; w < n.weights.size(); w++) {
                        if (index >= flat.size()) {
                            throw std::out_of_range("Not enough parameters provided to set_param.");
                        }
                        n.weights[w] = flat[index++];  // post-increment: reads index, then increments
                    }
                }
            }
        }

        // Runs the full forward pass through all layers sequentially.
        // Input: observation vector of length input_size.
        // Output: action vector of length output_size, each element in (-1, 1).
        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> current = inputs;
            for (layer& l : this->layers) {
                current = l.forward(current);
            }
            return current;
        }
};