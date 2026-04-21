# intra_layer_blockwise_cma_direct_policy_search.py
# CMA-ES optimisers over the full parameter vector of a C++ neural network.
# One CMA-ES instance per block — each block covers a contiguous group of neurons within a layer.
# All blocks are asked for candidates, the full network is assembled and evaluated,
# and the same fitness score is told to every block's CMA-ES instance.
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import numpy as np
import cma
import nn
from BaseCallback import BaseCallback


HIDDEN_LAYERS = (64, 64)  # Fixed architecture — matches SB3 MlpPolicy defaults for fair comparison.
SIGMA         = 0.05      # CMA-ES initial step size — pycma default
BLOCK_SIZE    = 1         # Default number of neurons per block


# =============================================================================
# intra_layer_blockwise_cma_direct_policy_search
# =============================================================================

class intra_layer_blockwise_cma_direct_policy_search:
    """
    CMA-ES optimisers over the full parameter vector of a C++ neural network.
    The parameter vector is partitioned into blocks (groups of neurons within each layer).
    Each block has its own CMA-ES instance with its own covariance matrix.
    All blocks are asked for a population of candidates each generation.
    The full network is assembled from one candidate per block, evaluated,
    and the same scalar fitness is told to every block's CMA-ES.
    Interface mirrors SB3: model.learn(total_timesteps, callback=callback)
    """

    def __init__(self, env, block_size: int = BLOCK_SIZE):

        self.env        = env
        self.block_size = block_size
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        # Build C++ neural network with blockwise parameter partitioning.
        self.nn = nn.NeuralNetwork(obs_dim, list(HIDDEN_LAYERS), act_dim, block_size)

        # Get the initial parameter blocks — one list per block.
        # init_blocks is a list of lists: [ block_0_params, block_1_params, ... ]
        init_blocks   = self.nn.get_param()
        self.n_blocks = len(init_blocks)

        # One CMA-ES instance per block.
        
        opts = {
            "CMA_diagonal": 0,
            "verbose":      -9,
            "popsize_factor": 0.5
        }
        self.es_list = [cma.CMAEvolutionStrategy(block, SIGMA, opts)for block in init_blocks]

        # Training state.
        self.global_steps = 0
        self.best_blocks  = None
        self.best_score   = -np.inf

    def predict(self, obs):
        """Forward pass through the network."""
        out = np.asarray(self.nn.forward(np.asarray(obs, dtype=np.float64).tolist()), dtype=np.float64)
        return out

    def _episode(self, blocks, callback: BaseCallback | None):
        """
        Loads a list of blocks into the network and runs one full episode.
        Returns total episode return.
        """
        self.nn.set_param(blocks)
        obs, _ = self.env.reset()
        ep_ret, ep_len, done = 0.0, 0, False

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done    = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
            self.global_steps += 1

        if callback is not None:
            callback.on_episode_end(ep_ret, ep_len)

        return ep_ret

    def learn(self, total_timesteps: int, callback: BaseCallback | None = None):
        """
        Runs blockwise CMA-ES until total_timesteps environment steps have been taken.

        Each generation:
          1. Ask each block's CMA-ES for a population of candidate vectors.
          2. For each candidate index i, assemble the full network by taking
             candidate i from every block's population.
          3. Evaluate the assembled network — one episode per candidate.
          4. Tell the same fitness score to every block's CMA-ES.
        """
        if callback is not None:
            callback.on_training_start()

        while self.global_steps < total_timesteps:

            # Step 1: ask each block's CMA-ES for its population.
            # all_solutions[b] = list of popsize candidate vectors for block b
            all_solutions = [es.ask() for es in self.es_list]
            popsize       = min(len(sols) for sols in all_solutions)

            losses = []

            # Step 2 & 3: assemble full network candidate-by-candidate and evaluate.
            for i in range(popsize):
                if self.global_steps >= total_timesteps:
                    break

                # Assemble one full set of blocks from candidate i of each block's population.
                blocks = [all_solutions[b][i] for b in range(self.n_blocks)]

                score = self._episode(blocks, callback=callback)
                losses.append(-score)  # CMA-ES minimises — negate reward

                if score > self.best_score:
                    self.best_score  = score
                    self.best_blocks = [np.array(blk) for blk in blocks]

            if len(losses) == 0:
                break

            # Step 4: tell the same losses to every block's CMA-ES.
            for b, es in enumerate(self.es_list):
                sols_b = all_solutions[b][:len(losses)]
                if len(losses) == popsize: # # only tell if full generation was evaluated
                    es.tell(sols_b, losses)

        # Load best found blocks into the network.
        if self.best_blocks is not None:
            self.nn.set_param(self.best_blocks)

        if callback is not None:
            callback.on_training_end()

        return self