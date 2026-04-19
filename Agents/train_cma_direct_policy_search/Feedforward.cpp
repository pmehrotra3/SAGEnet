// Feedforward.cpp
// A lightweight fully-connected feedforward neural network implementation in pure C++.
// Supports arbitrary depth and width, tanh activations for hidden layers and linear output layer,
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
        bool                use_tanh; // if true, applies tanh activation; if false, output is linear

        // Constructor — stores the neuron's bias and weight vector.
        neuron(double bias, std::vector<double> weights, bool use_tanh=true){
            this->bias    = bias;
            this->weights = weights;
            this->use_tanh = use_tanh; 
        }

        // Computes: tanh( dot(inputs, weights) + bias )
        double forward(const std::vector<double>& inputs) {
            double z = dot_product(inputs, this->weights) + this->bias;
            return use_tanh ? tanh_activation(z): z;
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
        layer(int size, int input_layer_dim, bool use_tanh=true) {
            for (int i = 0; i < size; i++) {
                double              bias = 0.0;
                std::vector<double> weights(input_layer_dim, 0.0);
                this->neurons.push_back(neuron(bias, weights, use_tanh));
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

        // Constructs a fully-connected network:
        //   input_size          → dimension of the observation vector
        //   hidden_layer_sizes  → e.g. {64, 64} for two hidden layers of 64 neurons each
        //   output_size         → dimension of the action vector
        //
        // All weights are initialised to 0.0 — use set_param() to load CMA-ES solutions.
        neural_network(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size) {
            int prev_size = input_size;

            // Hidden layers.
            for (int size : hidden_layer_sizes) {
                layers.push_back(layer(size, prev_size));
                prev_size = size;
            }

            // Output layer — linear (no activation), raw action values passed directly
            // to the environment, which clips them to the action space bounds.
            layers.push_back(layer(output_size, prev_size, false));
        }

        // Flattens all parameters (bias then weights, neuron by neuron, layer by layer)
        // into a single vector — the format expected by CMA-ES.
        std::vector<double> get_param() {
            std::vector<double> params;
            for (layer& l : this->layers) {
                for (neuron& n : l.neurons) {
                    params.push_back(n.bias);
                    for (double w : n.weights) {
                        params.push_back(w);
                    }
                }
            }
            return params;
        }

        // Loads a flat parameter vector (produced by CMA-ES) back into the network.
        // Order must match get_param(): bias then weights, neuron by neuron, layer by layer.
        void set_param(const std::vector<double>& new_params) {
            size_t index = 0;
            for (layer& l : this->layers) {
                for (neuron& n : l.neurons) {
                    if (index >= new_params.size()) {
                        throw std::out_of_range("Not enough parameters provided to set_param.");
                    }
                    n.bias = new_params[index++];            // post-increment: reads index, then increments
                    for (size_t w = 0; w < n.weights.size(); w++) {
                        if (index >= new_params.size()) {
                            throw std::out_of_range("Not enough parameters provided to set_param.");
                        }
                        n.weights[w] = new_params[index++];  // post-increment: reads index, then increments
                    }
                }
            }
        }

        // Runs the full forward pass through all layers sequentially.
        // Input: observation vector of length input_size.
        // Output: action vector of length output_size, each element unbounded (linear output layer).
        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> current = inputs;
            for (layer& l : this->layers) {
                current = l.forward(current);
            }
            return current;
        }
};