""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ACTIONS = ["STAY_OUT", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

@dataclass
class QLearningConfig:
    n_episodes: int = 12000
    alpha: float = 0.15
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8
    seed: int = 42

def epsilon_at(episode: int, cfg: QLearningConfig) -> float:
    decay_episodes = max(1, int(cfg.n_episodes * cfg.epsilon_decay_fraction))
    if episode >= decay_episodes:
        return cfg.epsilon_end
    frac = episode / decay_episodes
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

def _state_key(state: tuple) -> str:
    return "|".join(str(x) for x in state)

class TabularQAgent:
    ""

    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config or QLearningConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.q: dict[str, np.ndarray] = {}
        self.visit_counts: dict[str, int] = {}

    def q_values(self, state: tuple) -> np.ndarray:
        return self.q.get(_state_key(state), np.zeros(N_ACTIONS)).copy()

    def select_action(self, state: tuple, valid_actions: list, epsilon: float) -> str:
        ""
        if not valid_actions:
            raise ValueError("select_action called with an empty valid_actions list")
        if epsilon > 0 and self.rng.random() < epsilon:
            return valid_actions[self.rng.integers(len(valid_actions))]
        qv = self.q_values(state)
        valid_idx = [ACTION_INDEX[a] for a in valid_actions]
        best_local = int(np.argmax(qv[valid_idx]))
        return valid_actions[best_local]

    def greedy_action(self, state: tuple, valid_actions: list) -> str:
        ""
        return self.select_action(state, valid_actions, epsilon=0.0)

    def update(self, state: tuple, action: str, reward: float, next_state: tuple,
               done: bool, next_valid_actions: Optional[list] = None) -> None:
        ""
        key = _state_key(state)
        if key not in self.q:
            self.q[key] = np.zeros(N_ACTIONS)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

        if done:
            target = reward
        else:
            next_q = self.q_values(next_state)
            if next_valid_actions:
                mask = np.full(N_ACTIONS, -np.inf)
                for a in next_valid_actions:
                    mask[ACTION_INDEX[a]] = next_q[ACTION_INDEX[a]]
                best_next = float(np.max(mask))
            else:
                best_next = float(np.max(next_q))
            target = reward + self.config.gamma * best_next

        idx = ACTION_INDEX[action]
        self.q[key][idx] += self.config.alpha * (target - self.q[key][idx])

    def n_visited_states(self) -> int:
        return len(self.q)

    def table_size(self) -> int:
        return len(self.q)

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "config": asdict(self.config),
            "q_table": {k: v.tolist() for k, v in self.q.items()},
            "visit_counts": self.visit_counts,
            "actions": ACTIONS,
        }
        path.write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: Path) -> "TabularQAgent":
        payload = json.loads(Path(path).read_text())
        cfg = QLearningConfig(**payload["config"])
        agent = cls(cfg)
        agent.q = {k: np.array(v, dtype=float) for k, v in payload["q_table"].items()}
        agent.visit_counts = payload.get("visit_counts", {})
        return agent
