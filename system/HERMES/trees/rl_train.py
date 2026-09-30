"""
Trains the HERMES tabular Q-learning strategy agent on the SIMULATED
rl_env.HermesStrategyEnv (see that file's docstring for exactly what is
real vs synthetic in the environment) and saves the Q-table + training
config + per-episode return history.

Training is 100% simulated (parametric environment sampling real fitted
models over randomised circuits/compounds/conditions each episode) - it is
NOT fit to any specific historical race, and in particular does NOT touch
2025 data (the project's evaluation holdout, see evaluate.py). This is a
deliberate scope choice (see rl_env.py's module docstring, "WHY NOT REPLAY
REAL RACES"), not an oversight.

Usage (from the repo root):
    .\\.venv\\Scripts\\python.exe system\\HERMES\\trees\\rl_train.py
    .\\.venv\\Scripts\\python.exe system\\HERMES\\trees\\rl_train.py --episodes 12000 --seed 42
Outputs (next to this script, system/HERMES/trees/rl_artifacts/):
    qtable.json               trained Q-table + config (rl_agent.TabularQAgent.save format)
    training_curve.csv        per-episode: episode, epsilon, return, n_visited_states
    training_curve.png        convergence plot (rolling mean of episode return vs episode)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_agent import QLearningConfig, TabularQAgent, epsilon_at  # noqa: E402
from rl_env import HermesStrategyEnv  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "rl_artifacts"


def train(cfg: QLearningConfig, repo_root=None, verbose: bool = True) -> tuple[TabularQAgent, list[dict]]:
    env = HermesStrategyEnv(repo_root=repo_root, seed=cfg.seed)
    agent = TabularQAgent(cfg)
    history = []

    t0 = time.time()
    for ep in range(cfg.n_episodes):
        eps = epsilon_at(ep, cfg)
        state = env.reset()
        done = False
        ep_return = 0.0
        n_steps = 0
        last_info = None
        while not done:
            valid = env.valid_actions()
            action = agent.select_action(state, valid, eps)
            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions() if not done else None
            agent.update(state, action, reward, next_state, done, next_valid)
            ep_return += reward
            state = next_state
            n_steps += 1
            last_info = info
        history.append({"episode": ep, "epsilon": eps, "return": ep_return,
                         "n_steps": n_steps, "n_visited_states": agent.n_visited_states(),
                         # mandatory-compound compliance (2026-09-30 reward redesign) - set on the
                         # terminal step's info dict; None only if an episode had zero steps (can't happen).
                         "mandatory_compliant": last_info["mandatory_compliant"] if last_info else None})
        if verbose and (ep + 1) % max(1, cfg.n_episodes // 10) == 0:
            recent = history[-max(1, cfg.n_episodes // 20):]
            recent_return = np.mean([h["return"] for h in recent])
            recent_compliance = np.mean([bool(h["mandatory_compliant"]) for h in recent])
            print(f"  episode {ep + 1:>6}/{cfg.n_episodes}  epsilon={eps:.3f}  "
                  f"recent mean return={recent_return:+.2f}s  compliance={recent_compliance:.0%}  "
                  f"visited states={agent.n_visited_states()}")
    if verbose:
        print(f"[rl_train] training took {time.time() - t0:.1f}s for {cfg.n_episodes} episodes")
    return agent, history


def save_training_curve(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "epsilon", "return", "n_steps", "n_visited_states",
                                           "mandatory_compliant"])
        w.writeheader()
        for row in history:
            w.writerow(row)


def save_training_plot(history: list[dict], path: Path, window: int = 200) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rl_train] matplotlib not available - skipping training_curve.png (CSV still written)")
        return
    returns = np.array([h["return"] for h in history])
    episodes = np.array([h["episode"] for h in history])
    if len(returns) >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(returns, kernel, mode="valid")
        rolling_x = episodes[window - 1:]
    else:
        rolling, rolling_x = returns, episodes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(episodes, returns, alpha=0.15, color="tab:blue", label="episode return")
    ax1.plot(rolling_x, rolling, color="tab:blue", label=f"rolling mean ({window})")
    ax1.set_ylabel("episode return (s, higher/less-negative = better)")
    ax1.set_title("HERMES RL training: convergence")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(episodes, [h["epsilon"] for h in history], color="tab:orange")
    ax2.set_ylabel("epsilon")
    ax2.set_xlabel("episode")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the HERMES tabular Q-learning strategy agent")
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    cfg = QLearningConfig(n_episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
                           epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                           epsilon_decay_fraction=args.epsilon_decay_fraction, seed=args.seed)
    print(f"[rl_train] config: {cfg}")

    agent, history = train(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / "qtable.json")
    save_training_curve(history, out_dir / "training_curve.csv")
    save_training_plot(history, out_dir / "training_curve.png")

    last_n = max(1, cfg.n_episodes // 20)
    first_mean = np.mean([h["return"] for h in history[:last_n]])
    last_mean = np.mean([h["return"] for h in history[-last_n:]])
    first_compliance = np.mean([bool(h["mandatory_compliant"]) for h in history[:last_n]])
    last_compliance = np.mean([bool(h["mandatory_compliant"]) for h in history[-last_n:]])
    print(f"\n[rl_train] mean return, FIRST {last_n} episodes: {first_mean:+.2f}s  "
          f"(mandatory-compound compliance: {first_compliance:.0%})")
    print(f"[rl_train] mean return, LAST  {last_n} episodes: {last_mean:+.2f}s  "
          f"(mandatory-compound compliance: {last_compliance:.0%})")
    print(f"[rl_train] visited states: {agent.n_visited_states()}  (table size: {agent.table_size()})")
    print(f"[rl_train] saved: {out_dir / 'qtable.json'}, training_curve.csv, training_curve.png")


if __name__ == "__main__":
    main()
