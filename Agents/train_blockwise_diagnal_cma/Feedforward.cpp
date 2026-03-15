#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>


class neuron{

    private:

        double dot_product(const std::vector<double>& vector1, const std::vector<double>& vector2){

            if (vector1.size() != vector2.size()) {
                throw std::invalid_argument("Vectors must be of the same length for dot product.");
            }

            double sum = 0.0; 
            for (size_t i = 0; i < vector1.size(); i++) {
                sum += vector1[i] * vector2[i];
            }
            return sum;
        }

        double tanh_activation(double x) {

            return std::tanh(x);
        }

    public:

        double bias;
        std::vector<double> weights; 

        neuron(double bias, std::vector<double> weights) {
            this->bias = bias;
            this->weights = weights;
        }

        double* getBias() {
            return &this->bias; 
        }

        std::vector<double>* getWeights(){
            return &this-> weights; 
        }

        double forward(const std::vector<double>& inputs) {
            double z = dot_product(inputs, this->weights) + this->bias;
            return this->tanh_activation(z);
        }       
}; 

class layer{

    public:

        std::vector<neuron> neurons;

        int block_size_per_layer; 

        layer(int size, int input_layer_dim, int block_size_per_layer) {

            if (size <= 0) {
                throw std::invalid_argument("Layer size must be greater than 0.");
            }

            if (block_size_per_layer <= 0) {
                throw std::invalid_argument("Block size per layer must be greater than 0.");
            }

            if (block_size_per_layer > size) {
                throw std::invalid_argument("Block size per layer cannot be greater than the number of neurons in the layer.");
            }
            
            this->block_size_per_layer = block_size_per_layer;

            for (int i = 0; i < size; i++) {
                double bias = 0.0; 
                std::vector<double> weights(input_layer_dim, 0.0);
                this->neurons.push_back(neuron(bias, weights));
            }
        }

        std::vector<std::vector<neuron>> getNeurons() {
            std::vector<std::vector<neuron>> result;
            int i = 0;
            while (i < (int)this->neurons.size()) {
                std::vector<neuron> b;
                int g = i;
                while (g < i + this->block_size_per_layer && g < (int)this->neurons.size()) {
                    b.push_back(this->neurons[g]);
                    g = g + 1;
                }
                i = i + this->block_size_per_layer;
                if (!b.empty()) result.push_back(b);
            }
            return result; 
        }

        size_t get_total_num_Neurons() {
            return this->neurons.size(); 
        }

        size_t get_num_Blocks() {
            return (this->neurons.size()/ this->block_size_per_layer) + ((this->neurons.size() % this->block_size_per_layer) > 0 ? 1 : 0);
        }


        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> outputs;
            for (neuron& neuron : this->neurons) {
                outputs.push_back(neuron.forward(inputs));
            }
            return outputs;
        }
};

class neural_network{

    public:

        std::vector<layer> layers;

        neural_network(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size, int block_size_per_layer) {

            int prev_size = input_size;
            
            // Build hidden layers
            for (int size : hidden_layer_sizes) {
                layers.push_back(layer(size, prev_size, block_size_per_layer));
                prev_size = size;
            }

            // Build output layer
            layers.push_back(layer(output_size, prev_size, block_size_per_layer));

        }

        std::vector<layer>* getLayers() {
            return &this->layers; 
        }

        size_t get_num_Layers() {
            return this->layers.size(); 
        }

        std::vector<std::vector<neuron>> get_param() {
            std::vector<std::vector<neuron>> params;
            for (layer& layer : this->layers) {
                std::vector<std::vector<neuron>> blocks = layer.getNeurons();
                for (std::vector<neuron>& block : blocks) {
                    params.push_back(block);
                }
            }
            return params;
        }

        void set_param(const std::vector<double>& new_params) {
            size_t index = 0;
            for (layer& layer : this->layers) {
                for (neuron& neuron : layer.neurons) {
                    if (index >= new_params.size()) {
                        throw std::out_of_range("Not enough parameters provided to set_param.");
                    }
                    neuron.bias = new_params[index++];
                    for (size_t w = 0; w < neuron.weights.size(); w++) {
                        if (index >= new_params.size()) {
                            throw std::out_of_range("Not enough parameters provided to set_param.");
                        }
                        neuron.weights[w] = new_params[index++];
                    }
                }
            }
        }

        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> current_output = inputs;
            for (layer& layer : this->layers) {
                current_output = layer.forward(current_output);
            }
            return current_output;
        }


};