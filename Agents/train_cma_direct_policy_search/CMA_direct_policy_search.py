# CMA_direct_policy_search.py
# CMA-ES optimiser over the full parameter vector of a C++ neural network.
# A single global CMA-ES instance optimises all network weights simultaneously.
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import numpy as np
import cma
import nn
from BaseCallback import BaseCallback


HIDDEN_LAYERS = (64, 64)  # Fixed architecture — matches SB3 MlpPolicy defaults for fair comparison.
SIGMA = 0.05              # CMA-ES initial step size — pycma default


# =============================================================================
# CMA_direct_policy_search
# =============================================================================

class CMA_direct_policy_search:
    """
    CMA-ES optimiser over the full flattened parameter vector of a C++ neural network.
    A single CMA-ES instance maintains a full covariance matrix over all weights simultaneously.
    Interface mirrors SB3: model.learn(total_timesteps, callback=callback)
    """

    def __init__(self, env):
        
        self.env = env  # Gymnasium environment to train the agent on
        obs_dim = int(env.observation_space.shape[0]) # dimension of the observation vector
        act_dim = int(env.action_space.shape[0])      # dimension of the action vector

        # Build C++ neural network with fixed architecture.
        self.nn = nn.NeuralNetwork(obs_dim, list(HIDDEN_LAYERS), act_dim)

        # Initialise CMA-ES over the full flattened parameter vector.
        opts = {
            "CMA_diagonal": 0,    # full covariance matrix — no diagonal approximation
            "verbose":      -9,   # silence pycma console output
            "popsize_factor": 0.5
        }
        self.es = cma.CMAEvolutionStrategy(self.nn.get_param(), SIGMA, opts)
        
        # Training state — updated throughout learning.
        self.global_steps = 0
        self.best_params  = None
        self.best_score   = -np.inf

    def predict(self, obs):
        """Forward pass through the network. Output layer — linear (no activation),
        raw action values passed directly to the environment, which clips them to the action space bounds."""
        out = np.asarray(self.nn.forward(np.asarray(obs, dtype=np.float64).tolist()), dtype=np.float64)
        return out

    def _episode(self, params, callback: BaseCallback | None):
        """
        Evaluates a candidate solution by running one full episode.
        Calls callback.on_episode_end() at episode end if a callback is provided.
        Returns total episode return.
        """
        self.nn.set_param(params)
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
        Runs CMA-ES until total_timesteps environment steps have been taken.
        Calls callback lifecycle methods at training start, each episode end, and training end.
        """
        if callback is not None:
            callback.on_training_start()

        while self.global_steps < total_timesteps:
            solutions = self.es.ask()
            losses    = []

            for sol in solutions:
                if self.global_steps >= total_timesteps:
                    break

                score = self._episode(sol, callback=callback)
                losses.append(-score)  # CMA-ES minimises, so negate reward

                if score > self.best_score:
                    self.best_score  = score
                    self.best_params = sol

            # Need at least mu solutions to update the covariance matrix.
            if len(losses) < int(self.es.sp.weights.mu):
                break

            self.es.tell(solutions[:len(losses)], losses)

        # Load best found parameters into the network.
        if self.best_params is not None:
            self.nn.set_param(self.best_params)

        if callback is not None:
            callback.on_training_end()

        return self