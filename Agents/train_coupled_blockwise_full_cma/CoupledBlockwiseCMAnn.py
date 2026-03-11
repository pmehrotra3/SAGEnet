# CoupledBlockwiseCMAnn.py
# CMA-ES optimiser over the full parameter vector of a C++ neural network.
#
# Two-round algorithm per generation:
#
#   GENERATION 1 — Standard joint blockwise CMA-ES (Round 1 only)
#   ALL OTHER GENERATIONS — Conditioned sampling with fixed bias (Round 2 only):
#     - All non-input layers biased directly from L0 deviations via ridge
#     - Fixed BIAS_WEIGHT, no adaptation
#
# Developed with assistance from:
#   Claude  (Anthropic)  — https://www.anthropic.com

import numpy as np
import math
import cma
import nn
from BaseCallback import BaseCallback


HIDDEN_LAYERS  = (64, 64)
SIGMA          = 0.05
BLOCK_SIZE     = 1
BIAS_WEIGHT    = 0.3
RIDGE_ALPHA    = 1e-7

# SmoothedBasisDictionary hyperparameters
BASIS_RANK     = 8      # Number of basis vectors to maintain (replaces mu)
BASIS_ALPHA    = 0.75    # Exponential smoothing rate (0 = no update, 1 = replace)
REORTH_EVERY   = 100     # Re-orthogonalize every N updates for numerical stability


# =============================================================================
# SmoothedBasisDictionary — Dynamic Basis Tracking via Exponential Smoothing
# =============================================================================

class SmoothedBasisDictionary:
    """
    Maintains a fixed-rank orthonormal basis that evolves via exponential
    smoothing. Replaces the list-based BlockDictionary.

    Update rule:
        B[target] = (1 - alpha) * B[target] + alpha * delta_new
    where target is chosen to balance alignment (relevance) with least-used
    (diversity), preventing basis collapse into a single dominant direction.

    Periodic QR re-orthogonalization keeps the basis numerically stable and
    ensures the Ridge Gram matrix K = B @ B.T stays well-conditioned.
    """

    def __init__(self, dim: int, rank: int = BASIS_RANK,
                 alpha: float = BASIS_ALPHA, reorth_every: int = REORTH_EVERY):
        self.rank         = rank
        self.alpha        = alpha
        self.reorth_every = reorth_every
        self.dim          = dim

        self.B      = np.zeros((rank, dim))   # basis matrix  (rank × dim)
        self.counts = np.zeros(rank)           # per-component update counts
        self._ready         = False
        self.total_updates  = 0

    # ------------------------------------------------------------------

    def update(self, candidates: list, losses: list, old_mean: np.ndarray):
        """Ingest the best candidate from this generation and update the basis."""
        best_idx = int(np.argmin(losses))
        delta    = np.array(candidates[best_idx]) - old_mean
        norm     = np.linalg.norm(delta)
        if norm < 1e-8:
            return
        delta /= norm

        if not self._ready:
            # Fill basis slots sequentially until we have `rank` vectors
            slot = self.total_updates % self.rank
            self.B[slot] = delta
            self.total_updates += 1
            if self.total_updates >= self.rank:
                self._reorthogonalize()
                self._ready = True
        else:
            # Choose target slot: balance alignment (relevance) with
            # least-used (diversity) to avoid rich-get-richer collapse.
            alignments  = np.abs(self.B @ delta)          # (rank,)
            scores      = alignments / (self.counts + 1)  # favour under-used slots
            target_idx  = int(np.argmax(scores))

            # Exponential smoothing update
            self.B[target_idx] = (
                (1.0 - self.alpha) * self.B[target_idx] + self.alpha * delta
            )
            self.counts[target_idx] += 1
            self.total_updates      += 1

            # Periodic QR re-orthogonalization
            if self.total_updates % self.reorth_every == 0:
                self._reorthogonalize()

    def _reorthogonalize(self):
        """QR decomposition: makes B truly orthonormal, stabilising Ridge."""
        Q, _ = np.linalg.qr(self.B.T)   # Q is (dim × rank)
        self.B = Q.T                      # back to (rank × dim)

    @property
    def deltas(self) -> np.ndarray:
        """Return the current basis matrix (rank × dim)."""
        return self.B

    def is_ready(self) -> bool:
        return self._ready


# =============================================================================
# CoupledBlockwiseCMAnn
# =============================================================================

class CoupledBlockwiseCMAnn:

    def __init__(self, env, block_size: int = BLOCK_SIZE):
        self.env        = env
        self.block_size = block_size
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        self.nn       = nn.NeuralNetwork(obs_dim, list(HIDDEN_LAYERS), act_dim, block_size)
        init_blocks   = self.nn.get_param()
        self.n_blocks = len(init_blocks)

        opts = {
            "CMA_diagonal":   True,
            "verbose":        -9,
            "CMA_mirrors":    0,
            "popsize_factor": 0.5,
        }
        self.es_list = [
            cma.CMAEvolutionStrategy(block, SIGMA, opts)
            for block in init_blocks
        ]

        # Replace BlockDictionary with SmoothedBasisDictionary
        self.dicts = [
            SmoothedBasisDictionary(dim=len(init_blocks[b]))
            for b in range(self.n_blocks)
        ]

        # Layer structure
        layer_sizes           = list(HIDDEN_LAYERS) + [act_dim]
        self.blocks_per_layer = [int(math.ceil(s / block_size)) for s in layer_sizes]
        self.layer_ranges     = []
        start = 0
        for count in self.blocks_per_layer:
            self.layer_ranges.append(list(range(start, start + count)))
            start += count

        # Chain topology: each layer looks at the previous layer
        self.parent_map = {}
        for layer_idx, block_indices in enumerate(self.layer_ranges):
            for b in block_indices:
                self.parent_map[b] = self.layer_ranges[layer_idx - 1] if layer_idx > 0 else []

        self.global_steps = 0
        self.generation   = 0
        self.best_blocks  = None
        self.best_score   = -np.inf

    # -------------------------------------------------------------------------

    def predict(self, obs):
        return np.asarray(self.nn.forward(np.asarray(obs, dtype=np.float64).tolist()), dtype=np.float64)

    def _episode(self, blocks, callback):
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

    def _ridge_bias(self, parent_b: int, dev: np.ndarray, child_b: int) -> np.ndarray:
        """
        Ridge regression: map parent deviation → child bias direction.

        With SmoothedBasisDictionary the basis B is (rank × dim) and always
        orthonormal after re-orthogonalization, so K = B @ B.T is (rank × rank)
        — constant size regardless of history length.
        """
        pd        = self.dicts[parent_b]
        cd        = self.dicts[child_b]
        child_dim = len(self.es_list[child_b].mean)

        if not pd.is_ready() or not cd.is_ready():
            return np.zeros(child_dim)

        D_p = pd.deltas   # (rank × parent_dim)
        D_c = cd.deltas   # (rank × child_dim)

        # Gram matrix is now fixed (rank × rank) — fast and stable
        K   = D_p @ D_p.T + RIDGE_ALPHA * np.eye(pd.rank)
        rhs = D_p @ dev   # (rank,)

        try:
            coeffs = np.linalg.solve(K, rhs)   # (rank,)
        except np.linalg.LinAlgError:
            return np.zeros(child_dim)

        bias = D_c.T @ coeffs                  # (child_dim,)
        norm = np.linalg.norm(bias)
        return bias / norm if norm > 1e-8 else np.zeros(child_dim)

    def _joint_evaluate_and_tell(self, all_solutions, old_means, callback):
        popsize = min(len(s) for s in all_solutions)
        losses  = []

        for i in range(popsize):
            if self.global_steps >= self.total_timesteps:
                break
            blocks = [all_solutions[b][i] for b in range(self.n_blocks)]
            score  = self._episode(blocks, callback)
            losses.append(-score)

            if score > self.best_score:
                self.best_score  = score
                self.best_blocks = [np.array(blk) for blk in blocks]
                print(f"New best: {self.best_score:.2f} at step {self.global_steps}")

        if not losses:
            return [], False

        if len(losses) == popsize:
            for b, es in enumerate(self.es_list):
                es.tell(all_solutions[b][:len(losses)], losses)
                # SmoothedBasisDictionary.update takes the same signature
                self.dicts[b].update(
                    all_solutions[b][:len(losses)],
                    losses,
                    old_means[b]
                )

        return losses, True

    # -------------------------------------------------------------------------

    def learn(self, total_timesteps: int, callback: BaseCallback | None = None):
        self.total_timesteps = total_timesteps

        if callback is not None:
            callback.on_training_start()

        while self.global_steps < total_timesteps:

            if self.generation == 0:
                # ============================================================
                # GENERATION 1: standard joint blockwise CMA-ES (Round 1 only)
                # ============================================================
                old_means     = [np.array(es.mean) for es in self.es_list]
                all_solutions = [es.ask() for es in self.es_list]

                _, ok = self._joint_evaluate_and_tell(all_solutions, old_means, callback)
                if not ok or self.global_steps >= total_timesteps:
                    break

            else:
                # ============================================================
                # ALL OTHER GENERATIONS: conditioned sampling only (Round 2)
                # ============================================================
                conditioned = [None] * self.n_blocks
                l0_devs     = {}

                for layer_idx, block_indices in enumerate(self.layer_ranges):
                    for b in block_indices:
                        es      = self.es_list[b]
                        mean_b  = np.array(es.mean)
                        sols    = es.ask()

                        if layer_idx == 0:
                            biased = [np.array(s) for s in sols]
                        else:
                            biased = []
                            for i, s in enumerate(sols):
                                # 1. Raw direction of child's own random guess
                                raw_dev  = np.array(s) - mean_b
                                raw_dist = np.linalg.norm(raw_dev) + 1e-8
                                raw_dir  = raw_dev / raw_dist

                                # 2. Coordinated intent from parent(s) via Ridge
                                biases = [
                                    self._ridge_bias(pb, l0_devs[pb][i], b)
                                    for pb in self.parent_map[b]
                                    if pb in l0_devs and i < len(l0_devs[pb])
                                ]
                                intent_dir = np.mean(biases, axis=0) if biases else np.zeros_like(raw_dir)

                                intent_norm = np.linalg.norm(intent_dir)
                                if intent_norm > 1e-8:
                                    intent_dir /= intent_norm

                                # 3. Interpolate raw guess with parent intent
                                combined_dir  = (1.0 - BIAS_WEIGHT) * raw_dir + BIAS_WEIGHT * intent_dir
                                combined_dir /= (np.linalg.norm(combined_dir) + 1e-8)

                                # 4. Reconstruct with original step distance
                                biased.append(mean_b + raw_dist * combined_dir)

                        conditioned[b] = biased
                        # Normalised deviations passed forward to next layer
                        l0_devs[b] = [
                            (np.array(s) - mean_b) / (np.linalg.norm(np.array(s) - mean_b) + 1e-8)
                            for s in sols
                        ]

                old_means_r2 = [np.array(es.mean) for es in self.es_list]
                _, ok = self._joint_evaluate_and_tell(conditioned, old_means_r2, callback)
                if not ok:
                    break

            self.generation += 1

        if self.best_blocks is not None:
            self.nn.set_param(self.best_blocks)

        if callback is not None:
            callback.on_training_end()

        return self