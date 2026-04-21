"""
Main Training Script: Q-Learning vs SARSA on Cliff Walking
===========================================================
- Trains both agents for NUM_EPISODES (>=500)
- Saves Learning Curve (Graph) and Policy Heatmap (Style)
- Crash-safe: stops on error and reports the command + reason
"""

import sys
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from cliff_walking_env import CliffWalkingEnv
from agents import QLearningAgent, SarsaAgent

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
NUM_EPISODES   = 500
ALPHA          = 0.1
GAMMA          = 0.9
EPSILON        = 0.1
SMOOTH_WINDOW  = 20   # Rolling average window for learning curve

# Output paths
GRAPH_PATH = "Graph.png"
STYLE_PATH = "style.png"

# ─────────────────────────────────────────────
# Seaborn / Matplotlib global style
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.2)
COLORS = {
    "qlearning": "#E94560",   # vivid red-pink
    "sarsa":     "#0F3460",   # deep navy blue
    "smooth_q":  "#FF8A80",
    "smooth_s":  "#82B1FF",
}


def smooth(data, window):
    """Apply rolling mean for smoother curves."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────

def train_qlearning(env, agent, num_episodes):
    """Train Q-Learning agent and return per-episode total rewards."""
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return rewards


def train_sarsa(env, agent, num_episodes):
    """Train SARSA agent and return per-episode total rewards."""
    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            total_reward += reward
        rewards.append(total_reward)
    return rewards


# ─────────────────────────────────────────────
# Visualization: Learning Curve (Graph.png)
# ─────────────────────────────────────────────

def plot_learning_curve(q_rewards, s_rewards, smooth_window, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#16213E')

    episodes = np.arange(1, len(q_rewards) + 1)
    smooth_q = smooth(q_rewards, smooth_window)
    smooth_s = smooth(s_rewards, smooth_window)
    smooth_x = np.arange(smooth_window, len(q_rewards) + 1)

    # Raw (faded)
    ax.plot(episodes, q_rewards, color=COLORS["qlearning"], alpha=0.2, linewidth=0.8)
    ax.plot(episodes, s_rewards, color=COLORS["sarsa"],     alpha=0.2, linewidth=0.8, linestyle='--')

    # Smoothed
    ax.plot(smooth_x, smooth_q, color=COLORS["qlearning"], linewidth=2.5, label=f"Q-Learning (smoothed, w={smooth_window})")
    ax.plot(smooth_x, smooth_s, color=COLORS["sarsa"],     linewidth=2.5, linestyle='--', label=f"SARSA (smoothed, w={smooth_window})")

    ax.set_title("Q-Learning vs SARSA — Learning Curve\n(Cliff Walking, 4×12 Grid)", color='white', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel("Episode", color='white', fontsize=12)
    ax.set_ylabel("Total Reward per Episode", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(facecolor='#0F3460', edgecolor='#444', labelcolor='white', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3, color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved learning curve → {save_path}")


# ─────────────────────────────────────────────
# Visualization: Policy Heatmap (style.png)
# ─────────────────────────────────────────────

def extract_policy_grid(agent, env):
    """Return a (rows x cols) array of best-action values for heatmap."""
    rows, cols = env.get_grid_shape()
    policy = np.zeros((rows, cols))
    q_vals = np.zeros((rows, cols))
    action_symbols = {0: '^', 1: 'v', 2: '<', 3: '>'}

    annotations = []
    for r in range(rows):
        row_ann = []
        for c in range(cols):
            state = r * cols + c
            best_a = agent.get_best_action(state)
            q_val = agent.get_q(state, best_a)
            q_vals[r, c] = q_val
            row_ann.append(action_symbols[best_a])
        annotations.append(row_ann)

    return q_vals, annotations


def plot_policy_style(q_agent, s_agent, env, save_path):
    rows, cols = env.get_grid_shape()
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.patch.set_facecolor('#1A1A2E')

    titles = ["Q-Learning Policy (Off-policy)", "SARSA Policy (On-policy)"]
    agents = [q_agent, s_agent]
    cmaps = ["rocket", "mako"]

    for idx, (ax, agent, title, cmap) in enumerate(zip(axes, agents, titles, cmaps)):
        ax.set_facecolor('#16213E')
        q_vals, annotations = extract_policy_grid(agent, env)

        # Draw heatmap
        sns.heatmap(q_vals, ax=ax, cmap=cmap, annot=False,
                    linewidths=0.3, linecolor='#333344',
                    cbar=True, cbar_kws={'shrink': 0.7})

        # Overlay action arrows as text
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in env.cliff:
                    ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center',
                            fontsize=11, fontweight='bold', color='orange')
                elif pos == env.start:
                    ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='lime')
                elif pos == env.goal:
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='gold')
                else:
                    ax.text(c + 0.5, r + 0.5, annotations[r][c], ha='center',
                            va='center', fontsize=11, color='white')

        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Column", color='lightgray', fontsize=10)
        ax.set_ylabel("Row", color='lightgray', fontsize=10)
        ax.tick_params(colors='lightgray', labelsize=8)
        ax.collections[0].colorbar.ax.yaxis.set_tick_params(color='lightgray')
        plt.setp(ax.collections[0].colorbar.ax.yaxis.get_ticklabels(), color='lightgray')

    # Legend patches
    legend_elements = [
        mpatches.Patch(color='lime',   label='S = Start'),
        mpatches.Patch(color='gold',   label='G = Goal'),
        mpatches.Patch(color='orange', label='X = Cliff (Danger)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               facecolor='#0F3460', edgecolor='#444', labelcolor='white', fontsize=11,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Learned Policy Maps — Q-Learning vs SARSA", color='white',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved policy style map → {save_path}")


# ─────────────────────────────────────────────
# Main Execution with Crash Guard
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Q-Learning vs SARSA — Cliff Walking")
    print(f"  Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Episodes: {NUM_EPISODES} | α={ALPHA} | γ={GAMMA} | ε={EPSILON}")
    print("=" * 60)

    env = CliffWalkingEnv()

    # ── Train Q-Learning ──
    print("\n[1/2] Training Q-Learning...")
    q_env   = CliffWalkingEnv()
    q_agent = QLearningAgent(env.state_space, env.action_space, ALPHA, GAMMA, EPSILON)
    q_rewards = train_qlearning(q_env, q_agent, NUM_EPISODES)
    print(f"    Final 50-ep avg reward: {np.mean(q_rewards[-50:]):.2f}")

    # ── Train SARSA ──
    print("\n[2/2] Training SARSA...")
    s_env   = CliffWalkingEnv()
    s_agent = SarsaAgent(env.state_space, env.action_space, ALPHA, GAMMA, EPSILON)
    s_rewards = train_sarsa(s_env, s_agent, NUM_EPISODES)
    print(f"    Final 50-ep avg reward: {np.mean(s_rewards[-50:]):.2f}")

    # ── Plot ──
    print("\n[Plotting] Generating visualizations...")
    plot_learning_curve(q_rewards, s_rewards, SMOOTH_WINDOW, GRAPH_PATH)
    plot_policy_style(q_agent, s_agent, env, STYLE_PATH)

    print("\n✅ All done! Results saved to:")
    print(f"   • {GRAPH_PATH}")
    print(f"   • {STYLE_PATH}")

    return q_rewards, s_rewards, q_agent, s_agent


if __name__ == "__main__":
    CURRENT_COMMAND = "conda run -n py3.8 python main.py"
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌  CRASH DETECTED — 專案已停止執行")
        print("=" * 60)
        print(f"\n📌 執行指令：\n   {CURRENT_COMMAND}")
        print(f"\n💥 崩潰原因：\n   {type(e).__name__}: {e}")
        print("\n📋 完整 Traceback：")
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
