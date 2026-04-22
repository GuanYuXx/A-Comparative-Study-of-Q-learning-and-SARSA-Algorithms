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

    # ── Fix Y-axis to -500 ~ 0 for clear comparison ──
    ax.set_ylim(-500, 0)
    ax.set_yticks(range(-500, 1, 50))

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

# Action → (drow, dcol) for arrow direction
ACTION_DR = {0: -1, 1: 1, 2: 0, 3: 0}  # row delta (up=-1 in matrix coords)
ACTION_DC = {0: 0,  1: 0, 2: -1, 3: 1}  # col delta


def extract_policy_data(agent, env):
    """Return q_vals grid, and action arrays (U, V) for quiver arrows."""
    rows, cols = env.get_grid_shape()
    q_vals = np.zeros((rows, cols))
    U = np.zeros((rows, cols))  # col direction (x)
    V = np.zeros((rows, cols))  # row direction (y, inverted for plot)

    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            best_a = agent.get_best_action(state)
            q_vals[r, c] = agent.get_q(state, best_a)
            U[r, c] =  ACTION_DC[best_a]
            V[r, c] = -ACTION_DR[best_a]  # invert because y-axis is flipped in heatmap

    return q_vals, U, V


def get_optimal_path(agent, env, max_steps=200):
    """Greedy rollout from start to goal, returns list of (row, col)."""
    env_copy = CliffWalkingEnv()
    state_idx = env_copy.reset()
    path = [env_copy.state]
    for _ in range(max_steps):
        best_a = agent.get_best_action(state_idx)
        state_idx, _, done = env_copy.step(best_a)
        path.append(env_copy.state)
        if done:
            break
    return path


def plot_policy_style(q_agent, s_agent, env, save_path):
    rows, cols = env.get_grid_shape()

    # ── Distinct colormaps per agent for visual contrast ──
    # Q-Learning: warm red-orange (YlOrRd), SARSA: cool blue-teal (YlGnBu)
    AGENT_CFG = [
        {"title": "Q-Learning  (Off-policy)",  "cmap": "YlOrRd",  "arrow_color": "#FF4500",
         "path_color": "#FFD700", "agent": q_agent},
        {"title": "SARSA  (On-policy)",         "cmap": "YlGnBu", "arrow_color": "#00BFFF",
         "path_color": "#FFFACD", "agent": s_agent},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.patch.set_facecolor('#0D1117')

    for ax, cfg in zip(axes, AGENT_CFG):
        agent     = cfg["agent"]
        q_vals, U, V = extract_policy_data(agent, env)

        # ── Background heatmap ──
        im = ax.imshow(q_vals, cmap=cfg["cmap"], aspect='auto',
                       interpolation='nearest', alpha=0.85)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='lightgray')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='lightgray')
        cbar.set_label('Q-value', color='lightgray', fontsize=9)

        # ── Grid lines ──
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='#333344', linewidth=0.6)
        ax.tick_params(which='minor', length=0)

        # ── Quiver arrows (skip cliff & goal) ──
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in env.cliff or pos == env.goal:
                    continue
                ax.annotate("",
                    xy=(c + 0.5 + U[r, c] * 0.34, r + 0.5 - V[r, c] * 0.34),
                    xytext=(c + 0.5 - U[r, c] * 0.28, r + 0.5 + V[r, c] * 0.28),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=cfg["arrow_color"],
                        lw=1.8,
                        mutation_scale=16,
                    )
                )

        # ── Special cell labels ──
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in env.cliff:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#8B0000', alpha=0.7, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'CLIFF', ha='center', va='center',
                            fontsize=5.5, fontweight='bold', color='white', zorder=3)
                elif pos == env.start:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#006400', alpha=0.8, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'START', ha='center', va='center',
                            fontsize=7, fontweight='bold', color='lime', zorder=3)
                elif pos == env.goal:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#8B6914', alpha=0.8, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'GOAL', ha='center', va='center',
                            fontsize=7, fontweight='bold', color='gold', zorder=3)

        # ── Optimal path overlay ──
        opt_path = get_optimal_path(agent, env)
        if len(opt_path) > 1:
            path_cols = [p[1] + 0.5 for p in opt_path]
            path_rows = [p[0] + 0.5 for p in opt_path]
            ax.plot(path_cols, path_rows,
                    color=cfg["path_color"], linewidth=3.0, zorder=5,
                    marker='o', markersize=5, markerfacecolor=cfg["path_color"],
                    markeredgecolor='white', markeredgewidth=0.8,
                    label='Optimal Path')
            # Mark Start & Goal on path
            ax.plot(path_cols[0],  path_rows[0],  's', color='lime',
                    markersize=10, zorder=6, markeredgecolor='white', markeredgewidth=1)
            ax.plot(path_cols[-1], path_rows[-1], '*', color='gold',
                    markersize=14, zorder=6, markeredgecolor='white', markeredgewidth=0.8)

        # ── Axes style ──
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)   # top-left origin matches grid
        ax.set_xticks(np.arange(0.5, cols, 1))
        ax.set_yticks(np.arange(0.5, rows, 1))
        ax.set_xticklabels(range(cols), color='lightgray', fontsize=8)
        ax.set_yticklabels(range(rows), color='lightgray', fontsize=8)
        ax.set_title(cfg["title"], color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Column", color='lightgray', fontsize=10)
        ax.set_ylabel("Row",    color='lightgray', fontsize=10)
        ax.set_facecolor('#0D1117')
        ax.spines[:].set_color('#333')

    # ── Shared legend ──
    legend_elements = [
        mpatches.Patch(color='#006400', label='Start (S)'),
        mpatches.Patch(color='#8B6914', label='Goal (G)'),
        mpatches.Patch(color='#8B0000', label='Cliff (Danger)'),
        plt.Line2D([0], [0], color='#FFD700', lw=2.5, marker='o', markersize=5,
                   label='Optimal Path (Q-Learning)'),
        plt.Line2D([0], [0], color='#FFFACD', lw=2.5, marker='o', markersize=5,
                   linestyle='--', label='Optimal Path (SARSA)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               facecolor='#161B22', edgecolor='#444', labelcolor='white', fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Learned Policy Maps & Optimal Paths — Q-Learning vs SARSA",
                 color='white', fontsize=15, fontweight='bold', y=1.01)

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
