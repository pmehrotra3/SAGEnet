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

using namespace std;
using namespace Eigen;

// --- CMABlock Class (No changes needed) ---
class CMABlock {
public:
    int in_dim, block_size, weight_dim, param_dim;
    VectorXd mean;
    MatrixXd cov;
    double eps, max_cov;

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

// --- BlockedSageLayer Class (No changes needed) ---
class BlockedSageLayer {
public:
    int in_dim, out_dim;
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

// --- BlockedSageNet with Time-Limited Synchronized Search ---
class BlockedSageNet {
public:
    vector<BlockedSageLayer> layers;
    vector<pair<int, int>> initial_neuron_sequence;

    BlockedSageNet(int input_size, vector<int> hidden_sizes, int output_size, int block_size) {
        vector<int> sizes = {input_size};
        sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
        sizes.push_back(output_size);
        for (size_t i = 0; i < sizes.size() - 1; ++i)
            layers.emplace_back(BlockedSageLayer(sizes[i], sizes[i+1], block_size));
        
         for (size_t l=0; l<layers.size(); ++l) {
            for (size_t b=0; b<layers[l].blocks.size(); ++b) {
                initial_neuron_sequence.push_back({(int)l, (int)b});
            }
        }
    }

    double forward(const VectorXd &x, const vector<MatrixXd> &Ws, const vector<VectorXd> &Bs) const {
        VectorXd out = x;
        for (size_t i = 0; i < Ws.size(); ++i)
            out = (Ws[i] * out + Bs[i]).array().tanh();
        return out[0];
    }

    double loss(const vector<VectorXd> &x_data, const vector<double> &y_data, const vector<MatrixXd> &Ws, const vector<VectorXd> &Bs) const {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < x_data.size(); ++i) {
            double pred = forward(x_data[i], Ws, Bs);
            sum += pow(pred - y_data[i], 2);
        }
        return sum / x_data.size();
    }
    
    void train(const vector<VectorXd>& x_data, const vector<double>& y_data, 
               int epochs = 1000, int trials_per_neuron = 30, int top_k_local = 5, long long time_limit_ms = 50) {
        
        vector<MatrixXd> best_W(layers.size());
        vector<VectorXd> best_B(layers.size());
        
        for(size_t i = 0; i < layers.size(); ++i) {
            best_W[i].resize(layers[i].out_dim, layers[i].in_dim);
            best_B[i].resize(layers[i].out_dim);
            int row_start = 0;
            for(size_t j=0; j<layers[i].blocks.size(); ++j){
                auto& block = layers[i].blocks[j];
                auto [W,b] = block.sample();
                best_W[i].block(row_start, 0, block.block_size, block.in_dim) = W;
                best_B[i].segment(row_start, block.block_size) = b;
                row_start += block.block_size;
            }
        }
        double best_loss = loss(x_data, y_data, best_W, best_B);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            using Proposal = tuple<double, pair<int, int>, pair<MatrixXd, VectorXd>>;
            vector<Proposal> proposals;
            vector<vector<Proposal>> thread_private_proposals(omp_get_max_threads());
            
            auto epoch_start_time = chrono::high_resolution_clock::now();

            // --- Phase 1: Timed Parallel Local Search ---
            #pragma omp parallel for
            for(size_t i = 0; i < initial_neuron_sequence.size(); ++i) {
                const auto& neuron_idx = initial_neuron_sequence[i];
                int l_idx = neuron_idx.first;
                int b_idx = neuron_idx.second;
                auto& block = layers[l_idx].blocks[b_idx];
                vector<pair<pair<MatrixXd, VectorXd>, double>> block_trials;

                for (int t = 0; t < trials_per_neuron; ++t) {
                    // --- NEW TIME LIMIT CHECK ---
                    auto current_time = chrono::high_resolution_clock::now();
                    if (chrono::duration_cast<chrono::milliseconds>(current_time - epoch_start_time).count() > time_limit_ms) {
                        break; // Time's up for this epoch's search phase
                    }

                    auto [W_trial, b_trial] = block.sample();
                    vector<MatrixXd> temp_W = best_W;
                    vector<VectorXd> temp_B = best_B;
                    
                    int row_start = 0;
                    for(size_t k = 0; k < b_idx; ++k) row_start += layers[l_idx].blocks[k].block_size;
                    temp_W[l_idx].block(row_start, 0, block.block_size, block.in_dim) = W_trial;
                    temp_B[l_idx].segment(row_start, block.block_size) = b_trial;
                    
                    double trial_loss = loss(x_data, y_data, temp_W, temp_B);
                    block_trials.push_back({{W_trial, b_trial}, trial_loss});
                }

                if (!block_trials.empty()) {
                    sort(block_trials.begin(), block_trials.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
                    
                    if (block_trials[0].second < best_loss) {
                        thread_private_proposals[omp_get_thread_num()].emplace_back(block_trials[0].second, neuron_idx, block_trials[0].first);
                    }

                    vector<pair<pair<MatrixXd, VectorXd>, double>> top_k_for_update;
                    for(int k=0; k<min((int)block_trials.size(), top_k_local); ++k){
                         top_k_for_update.push_back(block_trials[k]);
                    }
                    block.apply_cma_update(top_k_for_update); 
                }
            }
            
            // --- Phase 2: The "Join and Compare" ---
            for(const auto& vec : thread_private_proposals) {
                proposals.insert(proposals.end(), vec.begin(), vec.end());
            }

            if (!proposals.empty()) {
                sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b){ return get<0>(a) < get<0>(b); });
                const auto& best_proposal = proposals[0];

                // --- Phase 3: The Global Commit ---
                best_loss = get<0>(best_proposal);
                auto [l_idx, b_idx] = get<1>(best_proposal);
                const auto& [W_best, B_best] = get<2>(best_proposal);
                
                int row_start = 0;
                for(size_t k = 0; k < b_idx; ++k) row_start += layers[l_idx].blocks[k].block_size;
                best_W[l_idx].block(row_start, 0, layers[l_idx].blocks[b_idx].block_size, layers[l_idx].in_dim) = W_best;
                best_B[l_idx].segment(row_start, layers[l_idx].blocks[b_idx].block_size) = B_best;
            }
            
            cout << "Epoch " << epoch + 1 << ": Loss = " << best_loss << endl;
        }
        cout << "\n✅ Final Loss with best network: " << best_loss << endl;
    }
};

// --- Main ---
int main() {
    vector<VectorXd> X;
    vector<double> y;
    int N = 100;
    for (int i = 0; i < N; ++i) {
        double xv = i / double(N - 1);
        VectorXd x(1); x << xv;
        X.push_back(x);
        y.push_back(log1p(exp(10.0 * (xv - 0.6))) / 10.0);
    }

    int block_size = 1; // Per-neuron
    BlockedSageNet model(1, {6,6,6,6}, 1, block_size); 

    auto start = chrono::high_resolution_clock::now();
    // epochs, trials_per_neuron, top_k_local_for_update, time_limit_ms
    model.train(X, y, 3000, 30, 5, 30); 
    auto end = chrono::high_resolution_clock::now();
    cout << "⏱️ Total training time: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " seconds" << endl;
}
