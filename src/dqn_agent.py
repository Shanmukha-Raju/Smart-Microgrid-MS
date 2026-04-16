"""
dqn_agent.py  —  DQN Agent for Microgrid Battery Control
==========================================================
Fixes applied vs v1:
  ① Removed Keras Lambda layer in dueling head (caused tf.placeholder error on
    older TF/Windows). Replaced with a custom DuelingHead layer.
  ② Training loop now uses tf.function-free pure numpy + eager execution to
    avoid Windows-specific graph-mode hanging.
  ③ Added explicit tf.config settings to suppress oneDNN / GPU warnings.
  ④ Replay buffer pre-fill now uses a fast vectorised loop.
  ⑤ Progress bar added so it's clear the script is alive.
"""

import os, sys, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN noise
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # suppress INFO/WARNING
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Ensure eager mode (should be default in TF2, but be explicit)
tf.compat.v1.disable_eager_execution.__module__  # just import check
# DO NOT call disable_eager_execution — that causes the placeholder bug

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_raw_data, engineer_features, MODEL_DIR
from rl_microgrid_env    import MicrogridEnv

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

AGENT_PATH = os.path.join(MODEL_DIR, "dqn_agent.keras")

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
STATE_DIM        = 7
N_ACTIONS        = 3
GAMMA            = 0.97
LR               = 3e-4
BATCH_SIZE       = 64
REPLAY_CAPACITY  = 20_000
MIN_REPLAY_SIZE  = 500
TARGET_UPDATE    = 200     # steps
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 0.9993
N_EPISODES       = 200
EPISODE_LEN      = 7 * 24  # 1 week per episode


# ─────────────────────────────────────────────
# CUSTOM DUELING HEAD (avoids tf.placeholder bug)
# ─────────────────────────────────────────────
class DuelingHead(layers.Layer):
    """Q(s,a) = V(s) + A(s,a) - mean(A(s,·))"""
    def __init__(self, n_actions: int, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = n_actions
        self.v_dense = layers.Dense(32, activation="relu")
        self.v_out   = layers.Dense(1)
        self.a_dense = layers.Dense(32, activation="relu")
        self.a_out   = layers.Dense(n_actions)

    def call(self, x):
        v = self.v_out(self.v_dense(x))                      # (B, 1)
        a = self.a_out(self.a_dense(x))                      # (B, n_actions)
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_actions": self.n_actions})
        return cfg


def build_dueling_dqn(state_dim: int = STATE_DIM,
                      n_actions: int  = N_ACTIONS) -> keras.Model:
    inp = keras.Input(shape=(state_dim,), name="state")
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dense(64,  activation="relu")(x)
    q   = DuelingHead(n_actions, name="dueling_head")(x)
    model = keras.Model(inputs=inp, outputs=q)
    return model


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self): return len(self.buf)


# ─────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.epsilon    = EPS_START
        self.step_count = 0
        self.online_net = build_dueling_dqn()
        self.target_net = build_dueling_dqn()
        self.target_net.set_weights(self.online_net.get_weights())
        self.optimizer  = keras.optimizers.Adam(LR, clipnorm=1.0)
        self.buffer     = ReplayBuffer(REPLAY_CAPACITY)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        q = self.online_net(state[np.newaxis], training=False).numpy()[0]
        return int(np.argmax(q))

    def learn(self) -> float:
        if len(self.buffer) < MIN_REPLAY_SIZE:
            return 0.0

        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)

        # Double DQN target
        next_a  = np.argmax(self.online_net(s2, training=False).numpy(), axis=1)
        tgt_q   = self.target_net(s2, training=False).numpy()
        next_q  = tgt_q[np.arange(BATCH_SIZE), next_a]
        targets = r + GAMMA * next_q * (1.0 - d)

        # Gradient step (pure eager — no graph compilation)
        with tf.GradientTape() as tape:
            q_all      = self.online_net(s, training=True)
            a_onehot   = tf.one_hot(a, N_ACTIONS)
            q_selected = tf.reduce_sum(q_all * a_onehot, axis=1)
            loss       = tf.reduce_mean(keras.losses.huber(targets, q_selected))

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.online_net.trainable_variables)
        )

        # Epsilon decay
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

        # Sync target net
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return float(loss)

    def save(self, path: str = AGENT_PATH):
        # Save with custom objects registered
        self.online_net.save(path)
        print(f"  DQN agent saved → {path}")

    def load(self, path: str = AGENT_PATH):
        self.online_net = keras.models.load_model(
            path, custom_objects={"DuelingHead": DuelingHead}
        )
        self.target_net.set_weights(self.online_net.get_weights())


# ─────────────────────────────────────────────
# LOAD PROFILES
# ─────────────────────────────────────────────
def load_profiles_from_dataset():
    df = load_raw_data()
    df = engineer_features(df)
    return (df["solar_pv_output"].values.astype(np.float32),
            df["grid_load_demand"].values.astype(np.float32))


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_dqn_agent():
    print("=" * 60)
    print("  DQN Agent — Microgrid Battery Control")
    print("=" * 60)

    solar_all, load_all = load_profiles_from_dataset()
    n_windows   = max(1, len(solar_all) // EPISODE_LEN)
    agent       = DQNAgent()
    ep_rewards, ep_losses = [], []

    # Pre-fill replay buffer with random policy (fast)
    print(f"  Pre-filling replay buffer ({MIN_REPLAY_SIZE} steps)...")
    s_win = solar_all[:EPISODE_LEN]
    l_win = load_all[:EPISODE_LEN]
    env   = MicrogridEnv(s_win, l_win)
    obs, _ = env.reset()
    filled = 0
    while filled < MIN_REPLAY_SIZE:
        a = env.action_space.sample()
        n_obs, rew, done, _, _ = env.step(a)
        agent.buffer.push(obs, a, rew, n_obs, float(done))
        obs = n_obs
        filled += 1
        if done:
            obs, _ = env.reset()
    print(f"  Buffer pre-filled. Starting training for {N_EPISODES} episodes...\n")

    for ep in range(N_EPISODES):
        # Pick random 7-day window
        idx   = np.random.randint(0, n_windows) * EPISODE_LEN
        idx   = min(idx, len(solar_all) - EPISODE_LEN)
        s_win = solar_all[idx: idx + EPISODE_LEN]
        l_win = load_all[idx:  idx + EPISODE_LEN]

        env.solar_profile = s_win
        env.load_profile  = l_win
        env.T             = EPISODE_LEN
        obs, _ = env.reset()

        ep_r, ep_l = 0.0, []
        for _ in range(EPISODE_LEN):
            action = agent.select_action(obs)
            n_obs, rew, done, _, _ = env.step(action)
            agent.buffer.push(obs, action, rew, n_obs, float(done))
            loss = agent.learn()
            if loss > 0:
                ep_l.append(loss)
            obs  = n_obs
            ep_r += rew
            if done:
                break

        ep_rewards.append(ep_r)
        ep_losses.append(float(np.mean(ep_l)) if ep_l else 0.0)

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(ep_rewards[-10:])
            bar   = "█" * int((ep + 1) / N_EPISODES * 30)
            bar   = bar.ljust(30)
            print(
                f"  [{bar}] Ep {ep+1:3d}/{N_EPISODES} | "
                f"AvgR(10): {avg_r:+8.2f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Buf: {len(agent.buffer):5d}"
            )

    agent.save(AGENT_PATH)
    _plot_training(ep_rewards, ep_losses)
    return agent


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_agent(agent: DQNAgent, solar: np.ndarray, load: np.ndarray):
    results = {}
    for label, policy in [("DQN Agent", "dqn"), ("Always Hold", "hold"), ("Naive Rule", "naive")]:
        env = MicrogridEnv(solar, load)
        obs, _ = env.reset()
        total_r, total_g = 0.0, 0.0
        for t in range(len(solar)):
            hour = t % 24
            if policy == "dqn":
                q = agent.online_net(obs[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(q))
            elif policy == "hold":
                action = 1
            else:
                if solar[t] > load[t] and env.soc < 0.85:  action = 0
                elif 17 <= hour <= 21 and env.soc > 0.2:    action = 2
                else:                                        action = 1
            obs, rew, done, _, info = env.step(action)
            total_r += rew
            total_g += info["grid_draw_kw"]
            if done: break
        results[label] = {"reward": total_r, "grid": total_g}

    print("\n──── EVALUATION (Last 7 Days) ──────────────────────")
    for lbl, v in results.items():
        print(f"  {lbl:15s} | Reward: {v['reward']:+10.2f} | Grid Draw: {v['grid']:8.1f} kWh")
    print("─────────────────────────────────────────────────────\n")

    # DQN should beat baseline — if not, train more episodes
    dqn_r   = results["DQN Agent"]["reward"]
    naive_r = results["Naive Rule"]["reward"]
    hold_r  = results["Always Hold"]["reward"]
    if dqn_r > naive_r:
        print("  ✅ DQN outperforms naive rule-based policy")
    else:
        print("  ⚠️  DQN hasn't converged yet — try increasing N_EPISODES")
    return results


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def _plot_training(rewards, losses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    win = min(20, len(rewards))
    sm  = np.convolve(rewards, np.ones(win)/win, mode="valid")
    axes[0].plot(rewards, alpha=0.3, color="steelblue")
    axes[0].plot(sm,      color="steelblue", linewidth=2, label=f"{win}-ep avg")
    axes[0].set_title("DQN — Episode Reward")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    axes[1].plot(losses, alpha=0.7, color="darkorange")
    axes[1].set_title("DQN — Huber Loss per Episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Mean Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "dqn_training.png"), dpi=120)
    print("  Plot saved → models/saved/dqn_training.png")
    plt.close()


if __name__ == "__main__":
    agent = train_dqn_agent()
    solar, load = load_profiles_from_dataset()
    evaluate_agent(agent, solar[-168:], load[-168:])