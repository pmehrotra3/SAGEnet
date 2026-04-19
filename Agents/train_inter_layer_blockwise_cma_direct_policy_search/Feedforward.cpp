// Feedforward.cpp
// Variant of the lightweight feedforward network for inter-layer blockwise CMA-ES.
//
// Every HIDDEN-layer neuron factors its weight vector and bias as elementwise/scalar products:
//     effective_weight = w_1 ⊙ w_2     (elementwise Hadamard product, both same length = fan-in)
//     effective_bias   = b_1 · b_2     (scalar product)
//     output           = tanh( (w_1 ⊙ w_2) · x + b_1 · b_2 )
//
// The output layer is unchanged: standard linear w·x + b, no activation.
//
// Parameters are partitioned for inter-layer blockwise CMA-ES. Let k = block_size / 2
// (block_size must be EVEN).
//
// Block order for (obs → H_1 → H_2 → ... → H_N → output):
//
//   FRONT : chunks of k neurons of H_1, contributing only (w_1, b_1).
//   MIDDLE: for each adjacent pair (H_m, H_{m+1}), chunks of k neurons from EACH side
//           combined into a paired block = (w_2, b_2) of H_m + (w_1, b_1) of H_{m+1}.
//   BACK  : chunks of k neurons of H_N, contributing only (w_2, b_2).
//   OUT   : chunks of k neurons of the output layer, contributing full (w, b).
//
// If any chunk at the end of a section has fewer than k neurons (layer size not divisible by k),
// those remaining neurons form one additional smaller block on their own.
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
// Two modes:
//   doubled = true  (hidden layer): stores w_1, w_2, b_1, b_2.
//       forward: tanh((w_1 ⊙ w_2) · x + b_1 · b_2)
//   doubled = false (output layer): stores w, b.
//       forward: w · x + b       (no activation)
 
class neuron {
 
    private:
 
        double tanh_activation(double x) { return std::tanh(x); }
 
    public:
 
        bool doubled;
 
        // Standard-mode parameters (doubled == false).
        double              bias;
        std::vector<double> weights;
 
        // Factored-mode parameters (doubled == true).
        double              bias1;
        double              bias2;
        std::vector<double> weights1;
        std::vector<double> weights2;
 
        // Standard-mode factory — output-layer neuron, w and b initialised to 0.0.
        static neuron make_standard(int input_dim) {
            neuron n;
            n.doubled = false;
            n.bias    = 0.0;
            n.weights = std::vector<double>(input_dim, 0.0);
            return n;
        }
 
        // Factored-mode factory — hidden-layer neuron, all four params initialised to 1.0
        // so the effective weight (w_1 ⊙ w_2) = 1 and effective bias (b_1 · b_2) = 1 at start.
        // CMA-ES sigma then perturbs them off the symmetric starting point.
        static neuron make_doubled(int input_dim) {
            neuron n;
            n.doubled  = true;
            n.bias1    = 1.0;
            n.bias2    = 1.0;
            n.weights1 = std::vector<double>(input_dim, 1.0);
            n.weights2 = std::vector<double>(input_dim, 1.0);
            return n;
        }
 
        // Forward pass dispatches on mode.
        double forward(const std::vector<double>& inputs) {
            if (doubled) {
                if (inputs.size() != weights1.size()) {
                    throw std::invalid_argument("Input size mismatch in doubled neuron.");
                }
                double z = 0.0;
                for (size_t i = 0; i < inputs.size(); i++) {
                    z += (weights1[i] * weights2[i]) * inputs[i];
                }
                z += bias1 * bias2;
                return tanh_activation(z);
            } else {
                if (inputs.size() != weights.size()) {
                    throw std::invalid_argument("Input size mismatch in standard neuron.");
                }
                double z = 0.0;
                for (size_t i = 0; i < inputs.size(); i++) {
                    z += weights[i] * inputs[i];
                }
                z += bias;
                return z;
            }
        }
};
 
 
// =============================================================================
// Layer
// =============================================================================
 
class layer {
 
    public:
 
        std::vector<neuron> neurons;
        bool                doubled;  // all neurons in this layer share the same mode
 
        layer(int size, int input_dim, bool doubled) {
            this->doubled = doubled;
            for (int i = 0; i < size; i++) {
                if (doubled) neurons.push_back(neuron::make_doubled(input_dim));
                else         neurons.push_back(neuron::make_standard(input_dim));
            }
        }
 
        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> outputs;
            outputs.reserve(neurons.size());
            for (neuron& n : neurons) outputs.push_back(n.forward(inputs));
            return outputs;
        }
};
 
 
// =============================================================================
// NeuralNetwork
// =============================================================================
 
class neural_network {
 
    public:
 
        std::vector<layer> layers;
        int                block_size;
        int                k;  // = block_size / 2
 
        neural_network(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size, int block_size) {
            if (block_size <= 0 || block_size % 2 != 0) {
                throw std::invalid_argument("block_size must be a positive even integer.");
            }
            this->block_size = block_size;
            this->k          = block_size / 2;
 
            int prev_size = input_size;
            for (int size : hidden_layer_sizes) {
                layers.push_back(layer(size, prev_size, /*doubled=*/true));
                prev_size = size;
            }
            layers.push_back(layer(output_size, prev_size, /*doubled=*/false));
        }
 
        int num_hidden() const { return (int)layers.size() - 1; }
 
        // Partitions parameters into blocks as described in the file header.
        std::vector<std::vector<double>> get_param() {
            std::vector<std::vector<double>> blocks;
            int H = num_hidden();
 
            if (H == 0) {
                // No hidden layers — chunk the output layer alone.
                chunk_output_layer(layers[0], blocks);
                return blocks;
            }
 
            // FRONT — (w_1, b_1) halves of H_1.
            chunk_hidden_half(layers[0], /*use_half_1=*/true, blocks);
 
            // MIDDLE — paired blocks between each consecutive pair of hidden layers.
            for (int m = 0; m + 1 < H; m++) {
                chunk_hidden_pair(layers[m], layers[m + 1], blocks);
            }
 
            // BACK — (w_2, b_2) halves of H_N.
            chunk_hidden_half(layers[H - 1], /*use_half_1=*/false, blocks);
 
            // OUT — chunks of the output layer.
            chunk_output_layer(layers[H], blocks);
 
            return blocks;
        }
 
        // Loads blocks back into the network. Must match get_param()'s order and block sizes.
        void set_param(const std::vector<std::vector<double>>& blocks) {
            size_t b = 0;
            int H = num_hidden();
 
            if (H == 0) {
                b = load_output_layer(layers[0], blocks, b);
                if (b != blocks.size()) throw std::out_of_range("set_param: leftover blocks.");
                return;
            }
 
            b = load_hidden_half(layers[0], /*use_half_1=*/true, blocks, b);
            for (int m = 0; m + 1 < H; m++) {
                b = load_hidden_pair(layers[m], layers[m + 1], blocks, b);
            }
            b = load_hidden_half(layers[H - 1], /*use_half_1=*/false, blocks, b);
            b = load_output_layer(layers[H], blocks, b);
 
            if (b != blocks.size()) throw std::out_of_range("set_param: leftover blocks.");
        }
 
        std::vector<double> forward(const std::vector<double>& inputs) {
            std::vector<double> current = inputs;
            for (layer& l : this->layers) current = l.forward(current);
            return current;
        }
 
    private:
 
        // -------------------- get_param helpers --------------------
 
        // One-sided half chunker. If use_half_1: pack (b_1, w_1). Else: pack (b_2, w_2).
        // Step in chunks of k. If the last chunk has <k neurons, it forms one smaller overflow block.
        void chunk_hidden_half(layer& l, bool use_half_1, std::vector<std::vector<double>>& blocks) {
            size_t N = l.neurons.size();
            for (size_t i = 0; i < N; i += (size_t)this->k) {
                size_t end = std::min(i + (size_t)this->k, N);
                std::vector<double> block;
                for (size_t j = i; j < end; j++) {
                    neuron& n = l.neurons[j];
                    double bval                        = use_half_1 ? n.bias1    : n.bias2;
                    const std::vector<double>& wvec    = use_half_1 ? n.weights1 : n.weights2;
                    block.push_back(bval);
                    for (double v : wvec) block.push_back(v);
                }
                blocks.push_back(block);
            }
        }
 
        // Paired chunker between two adjacent hidden layers a and b.
        // For i = 0, k, 2k, ... step in lockstep over BOTH layers:
        //   the block gets (b_2, w_2) of neurons i..i+k-1 in a, then (b_1, w_1) of same range in b.
        // Layers need not be the same size: iterate up to min(N_a, N_b) in full k-chunks.
        // Any leftover neurons on either side (beyond the min or in a non-multiple-of-k tail)
        // become ADDITIONAL one-sided overflow blocks appended after the paired run:
        //   first a's remaining (b_2, w_2) halves, then b's remaining (b_1, w_1) halves.
        void chunk_hidden_pair(layer& la, layer& lb, std::vector<std::vector<double>>& blocks) {
            size_t Na = la.neurons.size();
            size_t Nb = lb.neurons.size();
            size_t K  = (size_t)this->k;
            size_t Npair = std::min(Na, Nb);
 
            size_t i = 0;
            // Paired portion — both sides have enough neurons for a full k-chunk on each side.
            while (i + K <= Npair) {
                std::vector<double> block;
                // a side: (b_2, w_2) of neurons i..i+K-1
                for (size_t j = i; j < i + K; j++) {
                    neuron& n = la.neurons[j];
                    block.push_back(n.bias2);
                    for (double v : n.weights2) block.push_back(v);
                }
                // b side: (b_1, w_1) of neurons i..i+K-1
                for (size_t j = i; j < i + K; j++) {
                    neuron& n = lb.neurons[j];
                    block.push_back(n.bias1);
                    for (double v : n.weights1) block.push_back(v);
                }
                blocks.push_back(block);
                i += K;
            }
 
            // Overflow on the a side — remaining (b_2, w_2) halves of a, from i onward, if any.
            if (i < Na) {
                std::vector<double> block;
                for (size_t j = i; j < Na; j++) {
                    neuron& n = la.neurons[j];
                    block.push_back(n.bias2);
                    for (double v : n.weights2) block.push_back(v);
                }
                blocks.push_back(block);
            }
 
            // Overflow on the b side — remaining (b_1, w_1) halves of b, from i onward, if any.
            if (i < Nb) {
                std::vector<double> block;
                for (size_t j = i; j < Nb; j++) {
                    neuron& n = lb.neurons[j];
                    block.push_back(n.bias1);
                    for (double v : n.weights1) block.push_back(v);
                }
                blocks.push_back(block);
            }
        }
 
        // Output-layer chunker. k neurons per block, each contributing full (b, w).
        // Last block may be smaller (overflow).
        void chunk_output_layer(layer& l, std::vector<std::vector<double>>& blocks) {
            size_t N = l.neurons.size();
            for (size_t i = 0; i < N; i += (size_t)this->k) {
                size_t end = std::min(i + (size_t)this->k, N);
                std::vector<double> block;
                for (size_t j = i; j < end; j++) {
                    neuron& n = l.neurons[j];
                    block.push_back(n.bias);
                    for (double v : n.weights) block.push_back(v);
                }
                blocks.push_back(block);
            }
        }
 
        // -------------------- set_param helpers --------------------
        //
        // Each load_* mirrors the corresponding chunk_* exactly: it walks the layer(s)
        // in the same order and consumes blocks off the `blocks` list starting at index b,
        // returning the new block cursor.
 
        size_t load_hidden_half(layer& l, bool use_half_1, const std::vector<std::vector<double>>& blocks, size_t b) {
            size_t N = l.neurons.size();
            for (size_t i = 0; i < N; i += (size_t)this->k) {
                size_t end = std::min(i + (size_t)this->k, N);
                if (b >= blocks.size()) throw std::out_of_range("set_param: not enough blocks.");
                const std::vector<double>& block = blocks[b++];
                size_t idx = 0;
                for (size_t j = i; j < end; j++) {
                    neuron& n = l.neurons[j];
                    double&              bref = use_half_1 ? n.bias1    : n.bias2;
                    std::vector<double>& wref = use_half_1 ? n.weights1 : n.weights2;
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    bref = block[idx++];
                    for (size_t w = 0; w < wref.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        wref[w] = block[idx++];
                    }
                }
            }
            return b;
        }
 
        size_t load_hidden_pair(layer& la, layer& lb, const std::vector<std::vector<double>>& blocks, size_t b) {
            size_t Na = la.neurons.size();
            size_t Nb = lb.neurons.size();
            size_t K  = (size_t)this->k;
            size_t Npair = std::min(Na, Nb);
 
            size_t i = 0;
            while (i + K <= Npair) {
                if (b >= blocks.size()) throw std::out_of_range("set_param: not enough blocks.");
                const std::vector<double>& block = blocks[b++];
                size_t idx = 0;
 
                // a side: (b_2, w_2) of neurons i..i+K-1
                for (size_t j = i; j < i + K; j++) {
                    neuron& n = la.neurons[j];
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    n.bias2 = block[idx++];
                    for (size_t w = 0; w < n.weights2.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        n.weights2[w] = block[idx++];
                    }
                }
                // b side: (b_1, w_1) of neurons i..i+K-1
                for (size_t j = i; j < i + K; j++) {
                    neuron& n = lb.neurons[j];
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    n.bias1 = block[idx++];
                    for (size_t w = 0; w < n.weights1.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        n.weights1[w] = block[idx++];
                    }
                }
                i += K;
            }
 
            // Overflow on a side.
            if (i < Na) {
                if (b >= blocks.size()) throw std::out_of_range("set_param: not enough blocks.");
                const std::vector<double>& block = blocks[b++];
                size_t idx = 0;
                for (size_t j = i; j < Na; j++) {
                    neuron& n = la.neurons[j];
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    n.bias2 = block[idx++];
                    for (size_t w = 0; w < n.weights2.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        n.weights2[w] = block[idx++];
                    }
                }
            }
 
            // Overflow on b side.
            if (i < Nb) {
                if (b >= blocks.size()) throw std::out_of_range("set_param: not enough blocks.");
                const std::vector<double>& block = blocks[b++];
                size_t idx = 0;
                for (size_t j = i; j < Nb; j++) {
                    neuron& n = lb.neurons[j];
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    n.bias1 = block[idx++];
                    for (size_t w = 0; w < n.weights1.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        n.weights1[w] = block[idx++];
                    }
                }
            }
 
            return b;
        }
 
        size_t load_output_layer(layer& l, const std::vector<std::vector<double>>& blocks, size_t b) {
            size_t N = l.neurons.size();
            for (size_t i = 0; i < N; i += (size_t)this->k) {
                size_t end = std::min(i + (size_t)this->k, N);
                if (b >= blocks.size()) throw std::out_of_range("set_param: not enough blocks.");
                const std::vector<double>& block = blocks[b++];
                size_t idx = 0;
                for (size_t j = i; j < end; j++) {
                    neuron& n = l.neurons[j];
                    if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                    n.bias = block[idx++];
                    for (size_t w = 0; w < n.weights.size(); w++) {
                        if (idx >= block.size()) throw std::out_of_range("set_param: block too short.");
                        n.weights[w] = block[idx++];
                    }
                }
            }
            return b;
        }
};