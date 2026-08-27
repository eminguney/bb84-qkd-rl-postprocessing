"""
BB84 post-processing decision simulation: DQN, tabular Q-learning, threshold-rule,
and static baselines (Always-Send / Always-Retry / Always-Drop).

Reproduces the environment, state/action space, and reward structure described in
Section 4-6 of the manuscript "Reinforcement learning-based adaptive decision-making
for post-processing in BB84 quantum key distribution", and produces the numbers used
to fill Tables 2/3 and the new statistical-significance table.

Design choices not fully pinned down by the manuscript text are documented inline
with `# ASSUMPTION:` comments.
"""

import json
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

ACTIONS = ["send", "retry", "drop"]
MAX_STEPS = 5  # ASSUMPTION: episode is forced to Drop after 5 retries
NUM_QUBITS = 300  # per Sec 6.1
MAX_MSG_LEN = 1000.0  # normalization constant (training msg length is U(100,1000))

W1, W2, W3 = 1.0, 1.0, 0.5  # concrete instantiation of Eq. 1 weights


@dataclass
class BB84Env:
    rng: np.random.Generator
    mode: str = "train"  # train | clean | noisy | attacked | sparse
    fixed_msg_len: float = None

    msg_len: int = field(init=False, default=0)
    step_count: int = field(init=False, default=0)

    def _sample_conditions(self):
        """Draw (noise_level, eve_present) for the current attempt."""
        if self.mode == "train":
            noise = self.rng.uniform(0.0, 0.3)
            eve = self.rng.random() < 0.5
        elif self.mode == "clean":
            noise = self.rng.uniform(0.0, 0.05)
            eve = False
        elif self.mode == "noisy":
            noise = self.rng.uniform(0.15, 0.30)
            eve = False
        elif self.mode == "attacked":
            noise = self.rng.uniform(0.05, 0.15)
            eve = True
        elif self.mode == "sparse":
            noise = self.rng.uniform(0.10, 0.25)
            eve = self.rng.random() < 0.3
        else:
            raise ValueError(self.mode)
        return noise, eve

    def _generate_key(self):
        noise, eve = self._sample_conditions()
        base_eff = 0.5
        eff = base_eff * (1.0 - 0.6 * noise)
        qber = 0.02 + 0.6 * noise
        if eve:
            eff *= 0.55  # eavesdropping disturbs measurement -> fewer usable bits
            qber += 0.18 + 0.15 * self.rng.random()
        if self.mode == "sparse":
            eff *= 0.45  # ASSUMPTION: cumulative errors shrink usable key further
        variability = self.rng.uniform(0.85, 1.15)
        eff = max(0.0, eff * variability)
        # rare hardware-failure edge case -> near-empty key
        if self.rng.random() < 0.03:
            eff *= 0.1
        key_bits = max(0, int(round(NUM_QUBITS * eff)))
        qber = float(np.clip(qber, 0.0, 0.5))
        return key_bits, qber, eve

    def reset(self):
        self.msg_len = (
            int(self.fixed_msg_len)
            if self.fixed_msg_len is not None
            else int(self.rng.uniform(100, 1000))
        )
        self.step_count = 0
        self.key_bits, self.qber, self.eve = self._generate_key()
        return self._state()

    def _state(self):
        key_norm = float(np.clip(self.key_bits / NUM_QUBITS, 0.0, 1.0))
        msg_norm = float(np.clip(self.msg_len / MAX_MSG_LEN, 0.0, 1.0))
        return np.array([key_norm, self.qber, float(self.eve), msg_norm], dtype=np.float32)

    def step(self, action_idx):
        action = ACTIONS[action_idx]
        self.step_count += 1
        sufficient = self.key_bits >= self.msg_len
        done = False
        breach = False
        success = False

        if action == "send":
            done = True
            # ASSUMPTION (v2, retuned): breach risk scales with QBER rather than
            # jumping to a near-fixed high floor whenever Eve is merely suspected,
            # so that sending under low-QBER + Eve-suspected conditions is a
            # genuine (if risky) option rather than a dominated one.
            if self.eve:
                p_breach = float(np.clip(0.15 + 0.65 * self.qber, 0.0, 0.9))
            else:
                p_breach = float(np.clip(0.03 + 0.10 * self.qber, 0.0, 0.2))
            breach = self.rng.random() < p_breach
            success = sufficient and not breach
            eff_score = min(1.0, self.msg_len / max(1, self.key_bits)) if sufficient else 0.0
            sec_score = -0.7 if breach else (0.9 if not self.eve else 0.5)
            reward = W1 * sec_score + W2 * (eff_score if success else -0.3) - W3 * 0.0
        elif action == "retry":
            if self.step_count >= MAX_STEPS:
                done = True
                reward = W1 * 0.5 + W2 * 0.0 - W3 * 1.0
            else:
                done = False
                reward = W1 * 0.0 + W2 * (-0.05) - W3 * 0.3
                self.key_bits, self.qber, self.eve = self._generate_key()
        elif action == "drop":
            done = True
            reward = W1 * 0.55 + W2 * 0.0 - W3 * 0.1
        else:
            raise ValueError(action)

        info = {"success": success, "breach": breach, "eve": self.eve, "steps": self.step_count}
        return self._state(), reward, done, info


def run_episode(env: BB84Env, policy_fn):
    """policy_fn(state) -> action_idx. Returns dict of episode-level metrics."""
    state = env.reset()
    total_key = env.key_bits
    n_retries = 0
    while True:
        action_idx = policy_fn(state)
        if ACTIONS[action_idx] == "retry":
            n_retries += 1
        state, reward, done, info = env.step(action_idx)
        if done:
            break
    used_key = total_key if info.get("success") else 0
    key_generated = env.key_bits if not info.get("success") else total_key
    return {
        "success": bool(info.get("success", False)),
        "breach": bool(info.get("breach", False)),
        "eve": bool(info.get("eve", False)),
        "steps": info["steps"],
        "retries": n_retries,
        "kue": (env.msg_len / max(1, total_key)) if info.get("success") else 0.0,
    }


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, *transition):
        if len(self.buf) < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.choice(len(self.buf), size=min(batch_size, len(self.buf)), replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_dqn(seed, episodes=5000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = BB84Env(rng=rng, mode="train")

    qnet = QNet()
    target = QNet()
    target.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=0.001)
    buffer = ReplayBuffer(10000)

    eps = 1.0
    eps_min = 0.1
    eps_decay = (1.0 - eps_min) / (episodes * 0.6)
    gamma = 0.95
    batch_size = 64
    sync_every = 100
    step_count = 0

    for ep in range(episodes):
        state = env.reset()
        while True:
            if rng.random() < eps:
                action = rng.integers(0, 3)
            else:
                with torch.no_grad():
                    q = qnet(torch.tensor(state).unsqueeze(0))
                    action = int(torch.argmax(q, dim=1).item())
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            step_count += 1

            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size, rng)
                s_t = torch.tensor(s)
                a_t = torch.tensor(a, dtype=torch.long)
                r_t = torch.tensor(r)
                s2_t = torch.tensor(s2)
                d_t = torch.tensor(d)
                q_vals = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(s2_t).max(dim=1)[0]
                    q_target = r_t + gamma * q_next * (1 - d_t)
                loss = nn.functional.mse_loss(q_vals, q_target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if step_count % sync_every == 0:
                target.load_state_dict(qnet.state_dict())

            if done:
                break
        eps = max(eps_min, eps - eps_decay)

    return qnet


def dqn_policy(qnet):
    def fn(state):
        with torch.no_grad():
            q = qnet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())
    return fn


def discretize(state, bins=5):
    key_b = min(bins - 1, int(state[0] * bins))
    qber_b = min(bins - 1, int((state[1] / 0.5) * bins))
    eve_b = int(state[2])
    msg_b = min(bins - 1, int(state[3] * bins))
    return (key_b, qber_b, eve_b, msg_b)


def train_qlearning(seed, episodes=10000, bins=5):
    rng = np.random.default_rng(seed)
    env = BB84Env(rng=rng, mode="train")
    q_table = {}
    alpha = 0.1
    gamma = 0.95
    eps = 1.0
    eps_min = 0.1
    eps_decay = (1.0 - eps_min) / (episodes * 0.6)

    def get_q(s):
        if s not in q_table:
            q_table[s] = np.zeros(3)
        return q_table[s]

    for ep in range(episodes):
        state = env.reset()
        s_disc = discretize(state, bins)
        while True:
            if rng.random() < eps:
                action = rng.integers(0, 3)
            else:
                action = int(np.argmax(get_q(s_disc)))
            next_state, reward, done, _ = env.step(action)
            s2_disc = discretize(next_state, bins)
            q_sa = get_q(s_disc)
            q_s2 = get_q(s2_disc)
            q_sa[action] += alpha * (reward + gamma * np.max(q_s2) * (1 - done) - q_sa[action])
            s_disc = s2_disc
            if done:
                break
        eps = max(eps_min, eps - eps_decay)

    return q_table, bins


def qlearning_policy(q_table, bins):
    def fn(state):
        s_disc = discretize(state, bins)
        if s_disc not in q_table:
            return 1  # unseen state -> retry (cautious default)
        return int(np.argmax(q_table[s_disc]))
    return fn


def threshold_policy(state):
    key_norm, qber, eve, msg_norm = state
    if eve > 0.5:
        return 2  # drop
    if key_norm >= msg_norm and qber < 0.11:
        return 0  # send
    return 1  # retry (env forces drop after MAX_STEPS)


def always(action_idx):
    def fn(state):
        return action_idx
    return fn


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate(policy_fn, mode, seed, episodes=1000, msg_len=64.0):
    rng = np.random.default_rng(seed)
    env = BB84Env(rng=rng, mode=mode, fixed_msg_len=msg_len)
    n_success = 0
    n_breach = 0
    n_eve = 0
    n_eve_avoided = 0
    kue_sum = 0.0
    step_sum = 0
    for _ in range(episodes):
        m = run_episode(env, policy_fn)
        n_success += int(m["success"])
        kue_sum += m["kue"]
        step_sum += m["steps"]
        if m["eve"]:
            n_eve += 1
            if not m["breach"]:
                n_eve_avoided += 1
        if m["breach"]:
            n_breach += 1
    msr = n_success / episodes
    kue = kue_sum / episodes
    sbar = (n_eve_avoided / n_eve) if n_eve > 0 else 1.0
    ald = 0.7 * (step_sum / episodes)  # ASSUMPTION: 0.7s wall-clock cost per decision step
    return {"MSR": msr, "KUE": kue, "SBAR": sbar, "ALD": ald}


SCENARIOS = ["clean", "noisy", "attacked", "sparse"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    results = {}  # method -> seed -> scenario -> metrics
    methods = {}

    print("Training DQN and Q-learning agents across 5 seeds...")
    dqn_nets = {}
    qlearn_tables = {}
    for seed in SEEDS:
        print(f"  seed {seed}: training DQN...")
        dqn_nets[seed] = train_dqn(seed, episodes=5000)
        print(f"  seed {seed}: training tabular Q-learning...")
        qlearn_tables[seed] = train_qlearning(seed, episodes=10000)

    for seed in SEEDS:
        methods[("DQN", seed)] = dqn_policy(dqn_nets[seed])
        qtab, bins = qlearn_tables[seed]
        methods[("Q-learning", seed)] = qlearning_policy(qtab, bins)
        methods[("Threshold-Rule", seed)] = threshold_policy
        methods[("Always-Send", seed)] = always(0)
        methods[("Always-Retry", seed)] = always(1)
        methods[("Always-Drop", seed)] = always(2)

    print("Evaluating all methods across 4 scenarios x 5 seeds...")
    for (method, seed), policy in methods.items():
        results.setdefault(method, {}).setdefault(seed, {})
        for scenario in SCENARIOS:
            eval_seed = 10000 + seed  # fixed eval episode set per seed, shared across methods
            results[method][seed][scenario] = evaluate(policy, scenario, eval_seed, episodes=1000, msg_len=64.0)

    with open("simulation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved simulation/results.json")


if __name__ == "__main__":
    main()
