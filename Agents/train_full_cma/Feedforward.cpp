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

        layer(int size, int input_layer_dim) {
            for (int i = 0; i < size; i++) {
                double bias = 0.0; 
                std::vector<double> weights(input_layer_dim, 0.0);
                this->neurons.push_back(neuron(bias, weights));
            }
        }

        std::vector<neuron>* getNeurons() {
            return &this->neurons; 
        }

        size_t get_num_Neurons() {
            return this->neurons.size(); 
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

        neural_network(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size) {

            int prev_size = input_size;
            
            // Build hidden layers
            for (int size : hidden_layer_sizes) {
                layers.push_back(layer(size, prev_size));
                prev_size = size;
            }

            // Build output layer
            layers.push_back(layer(output_size, prev_size));

        }

        std::vector<layer>* getLayers() {
            return &this->layers; 
        }

        size_t get_num_Layers() {
            return this->layers.size(); 
        }

        std::vector<double> get_param() {
            std::vector<double> params;
            for (layer& layer : this->layers) {
                for (neuron& neuron : layer.neurons) {
                    params.push_back(neuron.bias);
                    for (double weight : neuron.weights) {
                        params.push_back(weight);
                    }
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