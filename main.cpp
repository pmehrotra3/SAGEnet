// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Agent for CartPole
// Pure Evolutionary RL: Networks are evaluated by running complete episodes
// No supervised learning - fitness = total episode reward

// Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <omp.h>  
#include <Eigen/Dense> 
#include <sw/redis++/redis++.h>  
#include <nlohmann/json.hpp>  

using namespace std;
using namespace Eigen;
using namespace sw::redis;
using json = nlohmann::json;

// --- CMABlock Class ---
// Represents one "block" of neurons in a layer
// Each block maintains its own Gaussian distribution (mean + covariance matrix)
// for sampling neural network weights and biases 
class CMABlock {
public:
    int in_dim;      // Number of input connections to this block
    int block_size;  // Number of neurons in this block
    int weight_dim;  // Total number of weights (in_dim * block_size)
    int param_dim;   // Total parameters (weights + biases)
    
    VectorXd mean;   // Mean vector of the Gaussian distribution
    MatrixXd cov;    // Covariance matrix of the Gaussian distribution
    double eps;      // Minimum eigenvalue to prevent numerical instability
    double max_cov;  // Maximum eigenvalue to prevent explosion

    // Constructor: Initialize a block with identity covariance and zero mean
    CMABlock(int input_dim, int num_neurons_in_block,
             double eps_ = 1e-6, double max_cov_ = 7.0)
        : in_dim(input_dim), block_size(num_neurons_in_block),
          eps(eps_), max_cov(max_cov_) {
        weight_dim = in_dim * block_size;
        param_dim  = weight_dim + block_size;  // weights + biases
        mean = VectorXd::Zero(param_dim);      // Start with zero mean
        cov  = MatrixXd::Identity(param_dim, param_dim);  // Start with identity covariance
    }

    // Sample a set of weights and biases from the current Gaussian distribution
    // Returns: (Weight matrix, Bias vector) for this block
    pair<MatrixXd, VectorXd> sample() {

        // Thread-safe random number generator

        thread_local mt19937 rng(random_device{}() + omp_get_thread_num());
        
        // Regularize covariance to ensure numerical stability

        MatrixXd reg_cov = regularize_cov();
        
        // Sample from standard normal distribution N(0,1)

        normal_distribution<> dist(0, 1);
        VectorXd z(param_dim);
        for (int i = 0; i < param_dim; ++i)
            z[i] = dist(rng);

        // Transform standard normal sample using mean and Cholesky decomposition
        // sample = mean + L * z, where L * L^T = covariance

        VectorXd sample_vec = mean + reg_cov.llt().matrixL() * z;
        
        // Split sampled parameters into weights and biases
        MatrixXd W_block = Map<MatrixXd>(sample_vec.head(weight_dim).data(),
                                         block_size, in_dim);
        VectorXd b_block = sample_vec.tail(block_size);
        return {W_block, b_block};
    }

    // Update the distribution based on top-performing samples (CMA-ES update rule)
    // top_k: The best performing parameter samples and their fitness scores
    // lr_modifier: Learning rate multiplier for this update

    void apply_cma_update(const vector<pair<pair<MatrixXd, VectorXd>, double>>& top_k, double lr_modifier = 1.0) {
        int k = top_k.size();
        if (k == 0) return;

        // Calculate weighted rank-based fitness shaping weights
        // Better performers (lower fitness value = higher reward) get higher weights

        vector<double> weights(k);
        double denom = 0.0;
        if (k > 1) {
            for (int i = 0; i < k; ++i) {
                weights[i] = log(k + 0.5) - log(i + 1);  // Log-linear weighting
                denom += weights[i];
            }
            // Normalize weights to sum to 1
            for (double &w : weights) w /= denom;
        } else {
            weights[0] = 1.0;
        }

        // Compute weighted mean update from top performers

        VectorXd mean_update = VectorXd::Zero(param_dim);
        for (int i = 0; i < k; ++i) {
            const auto &params = top_k[i].first;
            // Flatten weight matrix and bias vector into single parameter vector
            VectorXd flat(param_dim);
            flat << Map<const VectorXd>(params.first.data(), weight_dim), params.second;
            mean_update += weights[i] * (flat - mean);
        }
        VectorXd updated_mean = mean + lr_modifier * mean_update;

        // Compute new covariance from weighted scatter of top performers

        MatrixXd new_cov = MatrixXd::Zero(param_dim, param_dim);
        for (int i = 0; i < k; ++i) {
            const auto &params = top_k[i].first;
            VectorXd flat(param_dim);
            flat << Map<const VectorXd>(params.first.data(), weight_dim), params.second;
            VectorXd delta = flat - updated_mean;
            new_cov += weights[i] * delta * delta.transpose();  // Outer product
        }
        
        // Apply exponential moving average to covariance (smoothing)

        double cma_lr = 0.01 * lr_modifier;
        mean = updated_mean;
        cov  = (1.0 - cma_lr) * cov + cma_lr * new_cov;
        cov  = regularize_cov();  // Keep covariance numerically stable
    }

private:
    // Regularize covariance matrix by clamping eigenvalues
    // Prevents: too small eigenvalues (numerical issues) and too large (explosion)

    MatrixXd regularize_cov() {
        SelfAdjointEigenSolver<MatrixXd> eig(cov);
        // Clamp eigenvalues between eps and max_cov
        VectorXd eigvals = eig.eigenvalues().cwiseMax(eps).cwiseMin(max_cov);
        MatrixXd eigvecs = eig.eigenvectors();
        // Reconstruct covariance: C = V * D * V^T
        return eigvecs * eigvals.asDiagonal() * eigvecs.transpose();
    }
};

// --- BlockedSageLayer Class ---
// Represents one layer of a neural network, divided into independent CMA blocks
// Each block can evolve its parameters independently

class BlockedSageLayer {
public:
    int in_dim;   // Input dimension to this layer
    int out_dim;  // Output dimension from this layer
    vector<CMABlock> blocks;  // The layer is split into multiple blocks

    // Constructor: Divide the layer into blocks of approximately equal size
    BlockedSageLayer(int input_dim, int output_dim, int block_size)
        : in_dim(input_dim), out_dim(output_dim) {
        // Calculate how many blocks we need
        int num_blocks = (output_dim + block_size - 1) / block_size;
        for (int i = 0; i < num_blocks; ++i) {
            // Last block might be smaller if output_dim doesn't divide evenly
            int cur_block_size = (i == num_blocks - 1)
                                   ? (output_dim - i * block_size)
                                   : block_size;
            blocks.emplace_back(CMABlock(in_dim, cur_block_size));
        }
    }
};

// --- CompleteNetworkSample Struct ---
// Holds a complete network configuration with all weights and biases
// Used during global sampling to evaluate entire networks

struct CompleteNetworkSample {
    vector<MatrixXd> W;  // Weight matrices for all layers
    vector<VectorXd> B;  // Bias vectors for all layers
    double fitness;      // Fitness score (negative total reward, lower is better)
};

// --- BlockedSageNet Class ---
// A multi-layer neural network where each layer is divided into CMA blocks
// Uses pure evolutionary optimization - networks evaluated by running episodes

class BlockedSageNet {
public:
    vector<BlockedSageLayer> layers;  // All layers in the network
    
    // Current best network parameters (used as the agent's policy)
    vector<MatrixXd> best_W;
    vector<VectorXd> best_B;
    double best_fitness;  // Best fitness achieved so far
    
    // Redis connection for environment interaction
    Redis* redis_ptr;

    // Constructor: Build a network with given architecture
    // input_size: Dimension of input (e.g., 4 for CartPole state)
    // hidden_sizes: Sizes of hidden layers
    // output_size: Dimension of output (e.g., 1 for action value)
    // block_size: Number of neurons per CMA block

    BlockedSageNet(int input_size, vector<int> hidden_sizes, int output_size, int block_size) {
        // Construct full architecture: input -> hidden layers -> output
        vector<int> sizes = {input_size};
        sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
        sizes.push_back(output_size);
        
        // Create layers
        for (size_t i = 0; i < sizes.size() - 1; ++i)
            layers.emplace_back(BlockedSageLayer(sizes[i], sizes[i+1], block_size));
        
        // Initialize best network by sampling from each block
        best_W.resize(layers.size());
        best_B.resize(layers.size());
        best_fitness = 1e9;  // Very high (bad) initial fitness
        
        for(size_t i = 0; i < layers.size(); ++i) {
            best_W[i].resize(layers[i].out_dim, layers[i].in_dim);
            best_B[i].resize(layers[i].out_dim);
            int row_start = 0;
            
            // Sample initial parameters for each block
            for(size_t j = 0; j < layers[i].blocks.size(); ++j) {
                auto& block = layers[i].blocks[j];
                auto [W, b] = block.sample();
                best_W[i].block(row_start, 0, block.block_size, block.in_dim) = W;
                best_B[i].segment(row_start, block.block_size) = b;
                row_start += block.block_size;
            }
        }
        
        redis_ptr = nullptr;  // Will be set later
    }
    
    // Set Redis connection for environment interaction

    void set_redis(Redis* redis) {
        redis_ptr = redis;
    }

    // Forward pass: Compute network output for given input
    // x: Input vector (state)
    // Ws: Weight matrices for each layer
    // Bs: Bias vectors for each layer
    // Returns: Scalar output (action value)

    double forward(const VectorXd &x, const vector<MatrixXd> &Ws, const vector<VectorXd> &Bs) const {
        VectorXd out = x;
        // Pass through each layer with tanh activation
        for (size_t i = 0; i < Ws.size(); ++i)
            out = (Ws[i] * out + Bs[i]).array().tanh();
        return out[0];  // Return scalar output
    }

    // Evaluate a network by running a COMPLETE EPISODE in the CartPole environment
    // This is the TRUE FITNESS FUNCTION for evolutionary RL
    // W, B: Network parameters to evaluate
    // Returns: Negative total reward (lower is better for CMA-ES minimization)

    double evaluate_network(const vector<MatrixXd>& W, const vector<VectorXd>& B) {
        if (!redis_ptr) {
            cerr << "Error: Redis not connected!" << endl;
            return 1e9;  // Return very bad fitness
        }
        
        const string STATE_KEY = "cartpole:state";
        const string ACTION_KEY = "cartpole:action";
        const string EXPERIENCE_KEY = "cartpole:experience";
        
        // Wait for initial state from simulator
        auto state_data = redis_ptr->brpop(STATE_KEY, 5);  // 5 second timeout
        if (!state_data) {
            return 1e9;  // Timeout - return bad fitness
        }
        
        auto state_vec = json::parse(state_data->second).get<vector<double>>();
        VectorXd current_state = Map<VectorXd>(state_vec.data(), state_vec.size());
        
        double total_reward = 0.0;
        bool terminated = false;
        
        // Run episode with this network
        while (!terminated) {
            // Use network to choose action
            double output = forward(current_state, W, B);
            int action = (output > 0) ? 1 : 0;
            
            // Send action to simulator
            redis_ptr->lpush(ACTION_KEY, json(action).dump());
            
            // Receive experience
            auto exp_data = redis_ptr->brpop(EXPERIENCE_KEY, 5);
            if (!exp_data) {
                return 1e9;  // Timeout
            }
            
            auto exp_json = json::parse(exp_data->second);
            double reward = exp_json["reward"].get<double>();
            terminated = exp_json["terminated"].get<bool>();
            
            total_reward += reward;
            
            if (!terminated) {
                auto next_state_vec = exp_json["next_state"].get<vector<double>>();
                current_state = Map<VectorXd>(next_state_vec.data(), next_state_vec.size());
            }
        }
        
        // Return negative reward (CMA-ES minimizes, we want to maximize reward)
        return -total_reward;
    }
    
    // Train the network using CMA-ES evolutionary optimization with GLOBAL SAMPLING
    // Each sampled network is evaluated by running a complete episode
    // num_trials: How many complete networks to sample and evaluate
    // top_k: How many best networks to use for updating CMA distributions

    void train(int num_trials = 20, int top_k = 5) {
        
        cout << "🔬 Starting CMA-ES training: sampling " << num_trials << " networks..." << endl;
        
        // --- Phase 1: Global Sampling and Episode Evaluation ---
        // Sample COMPLETE networks and evaluate each by running full episodes
        vector<CompleteNetworkSample> complete_trials;
        
        for (int trial = 0; trial < num_trials; ++trial) {
            cout << "  Trial " << (trial + 1) << "/" << num_trials << ": ";
            
            // Sample ALL blocks to create a complete network
            CompleteNetworkSample sample;
            sample.W.resize(layers.size());
            sample.B.resize(layers.size());
            
            // For each layer, sample all its blocks
            for (size_t l = 0; l < layers.size(); ++l) {
                sample.W[l].resize(layers[l].out_dim, layers[l].in_dim);
                sample.B[l].resize(layers[l].out_dim);
                
                int row_start = 0;
                // Sample each block in this layer
                for (size_t b = 0; b < layers[l].blocks.size(); ++b) {
                    auto& block = layers[l].blocks[b];
                    auto [W_sample, b_sample] = block.sample();
                    
                    // Insert this block's parameters into the complete layer matrix
                    sample.W[l].block(row_start, 0, block.block_size, block.in_dim) = W_sample;
                    sample.B[l].segment(row_start, block.block_size) = b_sample;
                    row_start += block.block_size;
                }
            }
            
            // Evaluate this COMPLETE network by running an episode
            sample.fitness = evaluate_network(sample.W, sample.B);
            
            cout << "Reward = " << (-sample.fitness) << endl;
            
            complete_trials.push_back(sample);
        }
        
        // --- Phase 2: Select Top-K Complete Networks ---
        // Sort all sampled networks by fitness (lower fitness = higher reward = better)
        sort(complete_trials.begin(), complete_trials.end(), 
             [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
        
        // Keep only the top-K best networks
        int k = min(top_k, (int)complete_trials.size());
        vector<CompleteNetworkSample> top_k_networks(complete_trials.begin(), 
                                                      complete_trials.begin() + k);
        
        cout << "✨ Top-" << k << " networks selected. Best reward: " << (-top_k_networks[0].fitness) << endl;
        
        // --- Phase 3: Update Each Block's CMA Distribution ---
        // For each block, extract its parameters from the top-K winning networks
        // and update that block's CMA distribution to favor those parameter regions
        for (size_t l = 0; l < layers.size(); ++l) {
            for (size_t b = 0; b < layers[l].blocks.size(); ++b) {
                auto& block = layers[l].blocks[b];
                
                // Extract THIS block's parameters from ALL top-K networks
                vector<pair<pair<MatrixXd, VectorXd>, double>> block_params_from_top_k;
                
                // Calculate where this block starts in the layer
                int row_start = 0;
                for (size_t bb = 0; bb < b; ++bb) {
                    row_start += layers[l].blocks[bb].block_size;
                }
                
                // For each winning network, extract this block's weights
                for (const auto& net : top_k_networks) {
                    // Extract this block's weights from the complete network
                    MatrixXd W_this_block = net.W[l].block(row_start, 0, 
                                                           block.block_size, 
                                                           block.in_dim);
                    VectorXd B_this_block = net.B[l].segment(row_start, block.block_size);
                    
                    // Store with the network's fitness score
                    block_params_from_top_k.push_back({{W_this_block, B_this_block}, net.fitness});
                }
                
                // Update this block's CMA distribution based on what worked in winning networks
                block.apply_cma_update(block_params_from_top_k);
            }
        }
        
        // --- Phase 4: Update Best Network ---
        // Keep the best network found as the current policy
        if (top_k_networks[0].fitness < best_fitness) {
            best_W = top_k_networks[0].W;
            best_B = top_k_networks[0].B;
            best_fitness = top_k_networks[0].fitness;
            cout << "🎯 New best network! Reward: " << (-best_fitness) << endl;
        }
    }
};

// --- Experience Struct ---
// Stores one transition from the environment
struct Experience {
    VectorXd state;
    int action;
    double reward;
    VectorXd next_state;
    bool terminated;
};

int main() {
    // --- Agent Setup ---
    cout << "🤖 AGENT: Starting CMA-ES Evolutionary RL Agent..." << endl;
    
    // Connect to Redis for communication with the CartPole simulator
    auto redis = Redis("tcp://127.0.0.1:6379");
    cout << "✅ AGENT: Connected to Redis." << endl;

    // CartPole environment parameters
    int state_dim = 4;   // [cart position, cart velocity, pole angle, pole angular velocity]
    int action_dim = 1;  // Single output (interpreted as action 0 or 1)
    int block_size = 1;  // Number of neurons per CMA block

    // Create neural network agent
    // Architecture: 4 inputs -> [16, 16] hidden -> 1 output
    BlockedSageNet model(state_dim, {16, 16}, action_dim, block_size);
    model.set_redis(&redis);
    
    cout << "✅ AGENT: Model initialized. Ready to evolve!" << endl;

    // --- Main Evolutionary Loop ---
    int generation = 0;
    
    while (true) {
        generation++;
        cout << "\n" << string(60, '=') << endl;
        cout << "GENERATION " << generation << endl;
        cout << string(60, '=') << endl;
        
        // Run one generation of CMA-ES evolution
        // Sample 20 networks, evaluate each on episodes, keep top 5, update CMA
        model.train(30, 5);
        
        cout << "\n✅ Generation " << generation << " complete!" << endl;
    }

    return 0;
}