import os
import csv
import time
import numpy as np
import gymnasium as gym
import cma
import cppnn


class EpisodeLogger:
    def __init__(self):
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_end_steps = []

    def on_episode_end(self, ep_return: float, ep_length: int, global_step: int):
        self.episode_returns.append(float(ep_return))
        self.episode_lengths.append(int(ep_length))
        self.episode_end_steps.append(int(global_step))


def _space_info(env: gym.Env):
    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"Only Box observations supported right now, got {type(obs_space)}")
    if len(obs_space.shape) != 1:
        raise ValueError(f"Only 1D Box observations supported, got shape={obs_space.shape}")
    obs_dim = int(obs_space.shape[0])

    if isinstance(act_space, gym.spaces.Discrete):
        return obs_dim, "discrete", int(act_space.n)

    if isinstance(act_space, gym.spaces.Box):
        if len(act_space.shape) != 1:
            raise ValueError(f"Only 1D Box action supported, got shape={act_space.shape}")
        return obs_dim, "box", int(act_space.shape[0])

    raise TypeError(f"Unsupported action space type: {type(act_space)}")


class CMAESModel:
    """
    SB3-like:
      model = CMAESModel(env_name, hidden_layers=..., sigma=...)
      model.learn(total_timesteps, logger=...)
    """

    def __init__(
        self,
        env_name: str,
        hidden_layers=(64, 64),
        sigma: float = 0.5,
        seed: int | None = None,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        verbose: bool = False,
    ):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # spaces
        self.obs_dim, self.action_type, self.act_dim = _space_info(self.env)

        # IMPORTANT ORDER: (input_size, hidden_layers, output_size)
        self.nn = cppnn.NeuralNetwork(self.obs_dim, list(hidden_layers), self.act_dim)

        # CMA DEFAULTS (no opts dict)
        x0 = self.nn.get_param()
        
        opts = {
            "CMA_diagonal": 0,     # 0 means: never switch to diagonal mode
            "verbose": -9,     # silences console output
            "verb_disp": 0,    # no display
            "verb_log": 0,     # no log files from pycma
            "verb_time": False,
            "elitist": True,       # use elitist selection (keep best solution across generations)
        }

        # remove None keys (pycma doesn't like None values)
        opts = {k: v for k, v in opts.items() if v is not None}

        self.es = cma.CMAEvolutionStrategy(x0, float(sigma), opts)

        # eval + logging behavior
        self.seed = seed
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)
        self.verbose = bool(verbose)

        # state
        self.global_steps = 0
        self.best_params = None
        self.best_score = -np.inf

        # optional: seed env once at start (not required)
        if self.seed is not None:
            self.env.reset(seed=int(self.seed))

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def get_param(self):
        return self.nn.get_param()

    def set_param(self, p):
        self.nn.set_param(p)

    def predict(self, obs):
        x = np.asarray(obs, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.obs_dim:
            raise ValueError(f"Obs dim mismatch: got {x.shape[0]} expected {self.obs_dim}")

        out = np.asarray(self.nn.forward(x.tolist()), dtype=np.float64).reshape(-1)

        if self.action_type == "discrete":
            return int(np.argmax(out))

        # Box action
        low = self.env.action_space.low
        high = self.env.action_space.high
        return np.clip(out, low, high)

    def _reset_env_for_eval_episode(self, episode_index: int):
        if self.seed is None or not self.deterministic_eval:
            obs, _ = self.env.reset()
            return obs

        # deterministic but different per episode
        obs, _ = self.env.reset(seed=int(self.seed + 10_000 * episode_index))
        return obs

    def _rollout_one_episode(self, logger: EpisodeLogger | None, episode_index: int):
        obs = self._reset_env_for_eval_episode(episode_index)

        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1
            self.global_steps += 1

        if logger is not None:
            logger.on_episode_end(ep_ret, ep_len, self.global_steps)

        return ep_ret

    def _evaluate_params(self, params, logger: EpisodeLogger | None):
        self.set_param(params)
        rets = []
        for k in range(self.n_eval_episodes):
            rets.append(self._rollout_one_episode(logger=logger, episode_index=k))
        return float(np.mean(rets))

    def learn(self, total_timesteps: int, logger: EpisodeLogger | None = None):
        total_timesteps = int(total_timesteps)
        if total_timesteps <= 0:
            return self

        start = time.time()

        while self.global_steps < total_timesteps:
            solutions = self.es.ask()

            losses = []
            scores = []

            for sol in solutions:
                if self.global_steps >= total_timesteps:
                    break

                score = self._evaluate_params(sol, logger=logger)
                scores.append(score)
                losses.append(-score)  # CMA minimizes

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = sol

            if not losses:
                break
            
            mu = int(self.es.sp.weights.mu)
            lam_eval = len(losses)

            if lam_eval < mu:
                if self.verbose:
                    print(f"[CMA-ES] stopping: only {lam_eval} evals (< mu={mu}) at budget end.")
                break

            self.es.tell(solutions[:len(losses)], losses)

            if self.verbose:
                print(
                    f"[CMA-ES] steps={self.global_steps}/{total_timesteps} "
                    f"gen_best={float(np.max(scores)):.3f} "
                    f"gen_mean={float(np.mean(scores)):.3f} "
                    f"best_so_far={self.best_score:.3f}"
                )

        if self.best_params is not None:
            self.set_param(self.best_params)

        if self.verbose:
            dt = time.time() - start
            print(f"[CMA-ES] finished: steps={self.global_steps}, best_score={self.best_score:.3f}, wall={dt:.1f}s")

        return self


def append_final_performance_csv(
    csv_path: str,
    env_name: str,
    total_timesteps: int,
    logger: EpisodeLogger,
    window: int = 100,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    rets = logger.episode_returns
    last_k = rets[-window:] if len(rets) >= window else rets
    mean_last_k = float(np.mean(last_k)) if last_k else float("nan")
    std_last_k = float(np.std(last_k)) if last_k else float("nan")

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["env_name", "total_timesteps", "episodes_completed", f"mean_last_{window}", f"std_last_{window}"])
        w.writerow([env_name, int(total_timesteps), len(rets), mean_last_k, std_last_k])

    return mean_last_k, std_last_k