// Generalized CMA-ES Agent with Redis Communication (Episode-Based)
// Works with any Gymnasium environment via command-line argument
// Pure Evolutionary RL: Networks evaluated by complete episode returns
// Developed with assistance from Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google)

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <fstream>
#include <deque>
#include <omp.h>  
#include <Eigen/Dense> 
#include <sw/redis++/redis++.h>  
#include <nlohmann/json.hpp>  

using namespace std;
using namespace Eigen;
using namespace sw::redis;
using json = nlohmann::json;

// --- CMABlock Class ---
class CMABlock {
public:
    int in_dim;
    int block_size;
    int weight_dim;
    int param_dim;
    
    VectorXd mean;
    MatrixXd cov;
    double eps;
    double max_cov;

    CMABlock(int input_dim, int num_neurons_in_block,
             double eps_ = 1e-6, double max_cov_ = 7.0)
        : in_dim(input_dim), block_size(num_neurons_in_block),
          eps(eps_), max_cov(max_cov_) {
        weight_dim = in_dim * block_size;
        param_dim  = weight_dim + block_size;
        mean = VectorXd::Zero(param_dim);
        cov  = MatrixXd::Identity(param_dim, param_dim);
    }

    pair<MatrixXd, VectorXd> sample() {
        thread_local mt19937 rng(random_device{}() + omp_get_thread_num());
        MatrixXd reg_cov = regularize_cov();
        normal_distribution<> dist(0, 1);
        VectorXd z(param_dim);
        for (int i = 0; i < param_dim; ++i)
            z[i] = dist(rng);
        VectorXd sample_vec = mean + reg_cov.llt().matrixL() * z;
        MatrixXd W_block = Map<MatrixXd>(sample_vec.head(weight_dim).data(),
                                         block_size, in_dim);
        VectorXd b_block = sample_vec.tail(block_size);
        return {W_block, b_block};
    }

    void apply_cma_update(const vector<pair<pair<MatrixXd, VectorXd>, double>>& top_k, double lr_modifier = 1.0) {
        int k = top_k.size();
        if (k == 0) return;

        vector<double> weights(k);
        double denom = 0.0;
        if (k > 1) {
            for (int i = 0; i < k; ++i) {
                weights[i] = log(k + 0.5) - log(i + 1);
                denom += weights[i];
            }
            for (double &w : weights) w /= denom;
        } else {
            weights[0] = 1.0;
        }

        VectorXd mean_update = VectorXd::Zero(param_dim);
        for (int i = 0; i < k; ++i) {
            const auto &params = top_k[i].first;
            VectorXd flat(param_dim);
            flat << Map<const VectorXd>(params.first.data(), weight_dim), params.second;
            mean_update += weights[i] * (flat - mean);
        }
        VectorXd updated_mean = mean + lr_modifier * mean_update;

        MatrixXd new_cov = MatrixXd::Zero(param_dim, param_dim);
        for (int i = 0; i < k; ++i) {
            const auto &params = top_k[i].first;
            VectorXd flat(param_dim);
            flat << Map<const VectorXd>(params.first.data(), weight_dim), params.second;
            VectorXd delta = flat - updated_mean;
            new_cov += weights[i] * delta * delta.transpose();
        }
        
        double cma_lr = 0.01 * lr_modifier;
        mean = updated_mean;
        cov  = (1.0 - cma_lr) * cov + cma_lr * new_cov;
        cov  = regularize_cov();
    }

private:
    MatrixXd regularize_cov() {
        SelfAdjointEigenSolver<MatrixXd> eig(cov);
        VectorXd eigvals = eig.eigenvalues().cwiseMax(eps).cwiseMin(max_cov);
        MatrixXd eigvecs = eig.eigenvectors();
        return eigvecs * eigvals.asDiagonal() * eigvecs.transpose();
    }
};

// --- BlockedSageLayer Class ---
class BlockedSageLayer {
public:
    int in_dim;
    int out_dim;
    vector<CMABlock> blocks;

    BlockedSageLayer(int input_dim, int output_dim, int block_size)
        : in_dim(input_dim), out_dim(output_dim) {
        int num_blocks = (output_dim + block_size - 1) / block_size;
        for (int i = 0; i < num_blocks; ++i) {
            int cur_block_size = (i == num_blocks - 1)
                                   ? (output_dim - i * block_size)
                                   : block_size;
            blocks.emplace_back(CMABlock(in_dim, cur_block_size));
        }
    }
};

// --- Episode Network Sample ---
struct EpisodeNetworkSample {
    vector<MatrixXd> W;
    vector<VectorXd> B;
    double episode_return;
};

// --- CMA-ES Agent Class ---
class CMAAgent {
public:
    string env_name;
    string STATE_KEY;
    string ACTION_KEY;
    string EXPERIENCE_KEY;
    
    vector<BlockedSageLayer> layers;
    vector<MatrixXd> best_W;
    vector<VectorXd> best_B;
    double best_return;
    
    Redis* redis_ptr;
    int state_dim;
    int action_dim;
    bool is_discrete;
    
    deque<EpisodeNetworkSample> episode_buffer;
    const int BUFFER_SIZE = 20;
    const int TOP_K = 5;
    int episode_count = 0;
    int total_timesteps = 0;

    CMAAgent(const string& environment_name, int state_size, int action_size, 
             bool discrete, int block_size = 1)
        : env_name(environment_name), state_dim(state_size), action_dim(action_size),
          is_discrete(discrete) {
        
        STATE_KEY = env_name + ":state";
        ACTION_KEY = env_name + ":action";
        EXPERIENCE_KEY = env_name + ":experience";
        
        vector<int> sizes = {state_dim, 16, 16, action_dim};
        
        for (size_t i = 0; i < sizes.size() - 1; ++i)
            layers.emplace_back(BlockedSageLayer(sizes[i], sizes[i+1], block_size));
        
        best_W.resize(layers.size());
        best_B.resize(layers.size());
        best_return = -1e9;
        
        for(size_t i = 0; i < layers.size(); ++i) {
            best_W[i].resize(layers[i].out_dim, layers[i].in_dim);
            best_B[i].resize(layers[i].out_dim);
        }
        
        redis_ptr = nullptr;
    }
    
    void set_redis(Redis* redis) {
        redis_ptr = redis;
    }
    
    void sample_network(vector<MatrixXd>& W, vector<VectorXd>& B) {
        for(size_t l = 0; l < layers.size(); ++l) {
            W[l].resize(layers[l].out_dim, layers[l].in_dim);
            B[l].resize(layers[l].out_dim);
            
            int row_start = 0;
            for(size_t b = 0; b < layers[l].blocks.size(); ++b) {
                auto& block = layers[l].blocks[b];
                auto [W_sample, b_sample] = block.sample();
                
                W[l].block(row_start, 0, block.block_size, block.in_dim) = W_sample;
                B[l].segment(row_start, block.block_size) = b_sample;
                row_start += block.block_size;
            }
        }
    }

    VectorXd forward(const VectorXd &x, const vector<MatrixXd> &Ws, const vector<VectorXd> &Bs) const {
        VectorXd out = x;
        for (size_t i = 0; i < Ws.size(); ++i)
            out = (Ws[i] * out + Bs[i]).array().tanh();
        return out;
    }
    
    json select_action(const VectorXd& state, const vector<MatrixXd>& W, const vector<VectorXd>& B) {
        VectorXd output = forward(state, W, B);
        
        if (is_discrete) {
            int action = 0;
            if (action_dim == 1) {
                action = (output[0] > 0) ? 1 : 0;
            } else {
                output.maxCoeff(&action);
            }
            return json(action);
        } else {
            vector<double> action_vec(action_dim);
            for (int i = 0; i < action_dim; ++i)
                action_vec[i] = output[i];
            return json(action_vec);
        }
    }
    
    double evaluate_episode(const vector<MatrixXd>& W, const vector<VectorXd>& B) {
        if (!redis_ptr) {
            cerr << "❌ Error: Redis not connected!" << endl;
            return -1e9;
        }
        
        auto state_data = redis_ptr->brpop(STATE_KEY, 5);
        if (!state_data) {
            cerr << "❌ Timeout waiting for initial state" << endl;
            return -1e9;
        }
        
        auto state_vec = json::parse(state_data->second).get<vector<double>>();
        VectorXd current_state = Map<VectorXd>(state_vec.data(), state_vec.size());
        
        double episode_return = 0.0;
        bool done = false;
        
        while (!done) {
            json action_json = select_action(current_state, W, B);
            redis_ptr->lpush(ACTION_KEY, action_json.dump());
            
            auto exp_data = redis_ptr->brpop(EXPERIENCE_KEY, 5);
            if (!exp_data) {
                cerr << "❌ Timeout waiting for experience" << endl;
                return -1e9;
            }
            
            auto exp_json = json::parse(exp_data->second);
            double reward = exp_json["reward"].get<double>();
            done = exp_json["terminated"].get<bool>();
            auto next_state_vec = exp_json["next_state"].get<vector<double>>();
            
            episode_return += reward;
            total_timesteps++;
            
            if (!done) {
                redis_ptr->brpop(STATE_KEY, 5);
                current_state = Map<VectorXd>(next_state_vec.data(), next_state_vec.size());
            }
        }
        
        return episode_return;
    }
    
    void train_episode() {
        episode_count++;
        
        EpisodeNetworkSample sample;
        sample.W.resize(layers.size());
        sample.B.resize(layers.size());
        sample_network(sample.W, sample.B);
        
        sample.episode_return = evaluate_episode(sample.W, sample.B);
        
        episode_buffer.push_back(sample);
        if (episode_buffer.size() > BUFFER_SIZE) {
            episode_buffer.pop_front();
        }
        
        if (sample.episode_return > best_return) {
            best_W = sample.W;
            best_B = sample.B;
            best_return = sample.episode_return;
        }
        
        if (episode_count % 10 == 0) {
            cout << "Episode " << episode_count 
                 << " | Return: " << sample.episode_return
                 << " | Best: " << best_return
                 << " | Timesteps: " << total_timesteps << endl;
        }
        
        // CMA updates happen every episode (after buffer has 10+ episodes)
        if (episode_buffer.size() >= 10) {
            // Print once when CMA updates start
            if (episode_count == 10) {
                cout << "  ✅ CMA now updating after every episode" << endl;
            }
            update_cma_from_top_k();
        }
    }
    
    void update_cma_from_top_k() {
        vector<EpisodeNetworkSample> sorted(episode_buffer.begin(), episode_buffer.end());
        sort(sorted.begin(), sorted.end(),
             [](const auto& a, const auto& b) { return a.episode_return > b.episode_return; });
        
        int k = min(TOP_K, (int)sorted.size());
        vector<EpisodeNetworkSample> top_k(sorted.begin(), sorted.begin() + k);
        
        for (size_t l = 0; l < layers.size(); ++l) {
            for (size_t b = 0; b < layers[l].blocks.size(); ++b) {
                auto& block = layers[l].blocks[b];
                
                vector<pair<pair<MatrixXd, VectorXd>, double>> block_params;
                
                int row_start = 0;
                for (size_t bb = 0; bb < b; ++bb) {
                    row_start += layers[l].blocks[bb].block_size;
                }
                
                for (const auto& sample : top_k) {
                    MatrixXd W_block = sample.W[l].block(row_start, 0, 
                                                         block.block_size, 
                                                         block.in_dim);
                    VectorXd B_block = sample.B[l].segment(row_start, block.block_size);
                    
                    block_params.push_back({{W_block, B_block}, -sample.episode_return});
                }
                
                block.apply_cma_update(block_params);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <env_name> [--total_timesteps N]" << endl;
        cerr << "Example: " << argv[0] << " CartPole-v1 --total_timesteps 100000" << endl;
        return 1;
    }
    
    string env_name = argv[1];
    int total_timesteps = 100000;
    
    for (int i = 2; i < argc; ++i) {
        if (string(argv[i]) == "--total_timesteps" && i + 1 < argc) {
            total_timesteps = stoi(argv[i + 1]);
        }
    }
    
    cout << "🤖 CMA-ES Agent: Starting for environment '" << env_name << "'" << endl;
    cout << "   Total timesteps: " << total_timesteps << endl;
    cout << "   Episode buffer: 20 episodes (FIFO)" << endl;
    cout << "   CMA update: Every episode using top-5 from buffer" << endl;
    
    auto redis = Redis("tcp://127.0.0.1:6379");
    cout << "✅ CMA-ES Agent: Connected to Redis." << endl;

    int state_dim, action_dim;
    bool is_discrete;
    
    if (env_name == "CartPole-v1") {
        state_dim = 4;
        action_dim = 2;
        is_discrete = true;
    } else if (env_name == "Pendulum-v1") {
        state_dim = 3;
        action_dim = 1;
        is_discrete = false;
    } else if (env_name == "Acrobot-v1") {
        state_dim = 6;
        action_dim = 3;
        is_discrete = true;
    } else if (env_name == "MountainCar-v0") {
        state_dim = 2;
        action_dim = 3;
        is_discrete = true;
    } else {
        state_dim = 4;
        action_dim = 2;
        is_discrete = true;
        cout << "⚠️  Warning: Unknown environment, using default dimensions" << endl;
    }

    CMAAgent agent(env_name, state_dim, action_dim, is_discrete);
    agent.set_redis(&redis);
    
    cout << "✅ CMA-ES Agent: Initialized. Starting training..." << endl;
    cout << "   State dim: " << state_dim << ", Action dim: " << action_dim 
         << ", Discrete: " << (is_discrete ? "Yes" : "No") << endl;

    while (agent.total_timesteps < total_timesteps) {
        agent.train_episode();
    }
    
    cout << "\n🎉 Training complete! Ran " << agent.episode_count << " episodes (" 
         << agent.total_timesteps << " timesteps)" << endl;
    cout << "🏆 Best episode return: " << agent.best_return << endl;

    return 0;
}
