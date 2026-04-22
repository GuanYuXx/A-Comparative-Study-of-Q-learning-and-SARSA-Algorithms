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
GRAPH_PATH        = "Graph.png"
STYLE_POLICY_PATH = "style_policy.png"   # Fig 1: policy direction arrows
STYLE_PATH_PATH   = "style_path.png"     # Fig 2: optimal path on clean grid

# ─────────────────────────────────────────────
# Seaborn / Matplotlib global style
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.2)
COLORS = {
    "qlearning": "#FF4444",   # vivid red
    "sarsa":     "#00E5FF",   # electric cyan — visible on dark background
    "smooth_q":  "#FF4444",
    "smooth_s":  "#00E5FF",
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

    # ── Fix Y-axis to -300 ~ 0 ──
    ax.set_ylim(-300, 0)
    ax.set_yticks(range(-300, 1, 25))

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


def get_optimal_path(agent, env, max_steps=500):
    """
    Greedy rollout from start to goal.
    Returns list of (row, col) ONLY if the path actually reaches Goal.
    Returns None if the agent loops or falls off cliff without reaching Goal.
    """
    env_copy = CliffWalkingEnv()
    state_idx = env_copy.reset()
    path = [env_copy.state]
    for _ in range(max_steps):
        best_a = agent.get_best_action(state_idx)
        state_idx, _, done = env_copy.step(best_a)
        path.append(env_copy.state)
        if done:
            # Only return path if we actually reached the goal (not reset to start)
            if env_copy.state == env_copy.goal:
                return path
            else:
                return None  # fell into cliff, path invalid
    return None  # timed out — never reached goal


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
        agent = cfg["agent"]
        q_vals, U, V = extract_policy_data(agent, env)

        # ── Background heatmap: use extent so cell(r,c) is centered at (c+0.5, r+0.5) ──
        im = ax.imshow(q_vals, cmap=cfg["cmap"], aspect='auto',
                       interpolation='nearest', alpha=0.85,
                       extent=[0, cols, rows, 0])  # left, right, bottom, top
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='lightgray')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='lightgray')
        cbar.set_label('Q-value', color='lightgray', fontsize=9)

        # ── Grid lines (at integer boundaries, cells are 0~1, 1~2, etc.) ──
        for x in range(cols + 1):
            ax.axvline(x, color='#444455', linewidth=0.6, zorder=1)
        for y in range(rows + 1):
            ax.axhline(y, color='#444455', linewidth=0.6, zorder=1)

        # ── WHITE thick arrows centered in each cell (skip cliff & goal & start) ──
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in env.cliff or pos == env.goal or pos == env.start:
                    continue
                cx, cy = c + 0.5, r + 0.5   # cell center
                dx, dy = U[r, c] * 0.38, -V[r, c] * 0.38  # arrow tip offset
                ax.annotate("",
                    xy=(cx + dx, cy + dy),
                    xytext=(cx - dx, cy - dy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="white",
                        lw=2.2,
                        mutation_scale=22,
                    ),
                    zorder=4
                )

        # ── Special cell overlays ──
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos in env.cliff:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#8B0000', alpha=0.85, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'CLIFF', ha='center', va='center',
                            fontsize=5.5, fontweight='bold', color='white', zorder=3)
                elif pos == env.start:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#006400', alpha=0.9, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center',
                            fontsize=13, fontweight='bold', color='lime', zorder=3)
                elif pos == env.goal:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                               color='#8B6914', alpha=0.9, zorder=2))
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center',
                            fontsize=13, fontweight='bold', color='gold', zorder=3)

        # ── Optimal path overlay (only if path actually reaches Goal) ──
        opt_path = get_optimal_path(agent, env)
        if opt_path is not None:
            path_cols = [p[1] + 0.5 for p in opt_path]
            path_rows = [p[0] + 0.5 for p in opt_path]
            ax.plot(path_cols, path_rows,
                    color=cfg["path_color"], linewidth=3.2, zorder=6,
                    marker='o', markersize=6, markerfacecolor=cfg["path_color"],
                    markeredgecolor='white', markeredgewidth=1.0)
            ax.plot(path_cols[0],  path_rows[0],  's', color='lime',
                    markersize=11, zorder=7, markeredgecolor='white', markeredgewidth=1.2)
            ax.plot(path_cols[-1], path_rows[-1], '*', color='gold',
                    markersize=16, zorder=7, markeredgecolor='white', markeredgewidth=1.0)
            print(f"    [{cfg['title'].split('(')[0].strip()}] Optimal path length: {len(opt_path)} steps")
        else:
            print(f"    [{cfg['title'].split('(')[0].strip()}] WARNING: optimal path did not reach Goal — skipping overlay")

        # ── Axes style ──
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_xticks(np.arange(0.5, cols, 1))
        ax.set_yticks(np.arange(0.5, rows, 1))
        ax.set_xticklabels(range(cols), color='lightgray', fontsize=8)
        ax.set_yticklabels(range(rows), color='lightgray', fontsize=8)
        ax.set_title(cfg["title"], color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Column", color='lightgray', fontsize=10)
        ax.set_ylabel("Row",    color='lightgray', fontsize=10)
        ax.set_facecolor('#0D1117')
        ax.spines[:].set_color('#555')

    # ── Shared legend ──
    legend_elements = [
        mpatches.Patch(color='#006400', label='S = Start'),
        mpatches.Patch(color='#8B6914', label='G = Goal'),
        mpatches.Patch(color='#8B0000', label='Cliff'),
        plt.Line2D([0], [0], color='white', lw=2, label='White arrow = best action'),
        plt.Line2D([0], [0], color='#FFD700', lw=2.5, marker='o', markersize=5,
                   label='Optimal Path (Q-Learning)'),
        plt.Line2D([0], [0], color='#FFFACD', lw=2.5, marker='o', markersize=5,
                   label='Optimal Path (SARSA)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               facecolor='#161B22', edgecolor='#444', labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Learned Policy Maps & Optimal Paths — Q-Learning vs SARSA",
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved policy style map → {save_path}")


# ─────────────────────────────────────────────
# Fig 1: Policy Direction Map  (style_policy.png)
# White background, bold arrow per cell, no gradient
# ─────────────────────────────────────────────

ARROW_CHARS = {0: '^', 1: 'v', 2: '<', 3: '>'}   # ASCII fallback


def _draw_policy_grid(ax, agent, env, title, path_color):
    """Draw one policy grid panel: white bg, bold arrows, small path overlay."""
    rows, cols = env.get_grid_shape()
    ax.set_facecolor('white')
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)   # row 0 at top

    # ── Fill special cells first ──
    for r in range(rows):
        for c in range(cols):
            if (r, c) in env.cliff:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                             color='#AED6F1', zorder=1))

    # ── Grid lines ──
    for x in range(cols + 1):
        ax.axvline(x, color='black', linewidth=1.2, zorder=2)
    for y in range(rows + 1):
        ax.axhline(y, color='black', linewidth=1.2, zorder=2)

    # ── Cell content: big policy arrows ──
    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            cx, cy = c + 0.5, r + 0.5
            if pos in env.cliff:
                if c == (env.cols // 2):
                    ax.text(cx, cy, 'Cliff', ha='center', va='center',
                            fontsize=9, color='#1A5276', fontweight='bold', zorder=3)
            elif pos == env.start:
                ax.text(cx, cy, 'S', ha='center', va='center',
                        fontsize=13, color='black', fontweight='bold', zorder=3)
            elif pos == env.goal:
                ax.text(cx, cy, 'G', ha='center', va='center',
                        fontsize=13, color='black', fontweight='bold', zorder=3)
            else:
                state = r * cols + c
                best_a = agent.get_best_action(state)
                u = ACTION_DC[best_a]
                v = -ACTION_DR[best_a]
                ax.annotate("",
                    xy    =(cx + u * 0.30, cy - v * 0.30),
                    xytext=(cx - u * 0.30, cy + v * 0.30),
                    arrowprops=dict(arrowstyle="-|>", color='black',
                                   lw=2.2, mutation_scale=20),
                    zorder=4
                )

    # ── Optimal path overlay (smaller, colored) ──
    opt_path = get_optimal_path(agent, env)
    if opt_path is not None:
        px = [p[1] + 0.5 for p in opt_path]
        py = [p[0] + 0.5 for p in opt_path]
        ax.plot(px, py, color=path_color, linewidth=2.0, zorder=6,
                marker='o', markersize=4,
                markerfacecolor=path_color, markeredgecolor='white',
                markeredgewidth=0.6, alpha=0.85)
        ax.plot(px[0],  py[0],  's', color='limegreen',
                markersize=8, zorder=7, markeredgecolor='black', markeredgewidth=1.0)
        ax.plot(px[-1], py[-1], '*', color='gold',
                markersize=12, zorder=7, markeredgecolor='black', markeredgewidth=0.6)

    # ── Axes cosmetics ──
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold', color='black', pad=8)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)


def plot_policy_arrows(q_agent, s_agent, env, save_path):
    """Save style_policy.png: 1x2 subplots of policy direction maps."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    _draw_policy_grid(axes[0], q_agent, env,
                      "Q-Learning Policy  (Off-policy)", path_color='#E53935')
    _draw_policy_grid(axes[1], s_agent, env,
                      "SARSA Policy  (On-policy)",       path_color='#1565C0')

    # ── Legend for the path overlay ──
    legend_elements = [
        plt.Line2D([0], [0], color='#E53935', lw=2, marker='o', markersize=5,
                   label='Optimal Path (Q-Learning)'),
        plt.Line2D([0], [0], color='#1565C0', lw=2, marker='o', markersize=5,
                   label='Optimal Path (SARSA)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='limegreen',
                   markersize=8, markeredgecolor='black', label='S = Start'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=11, markeredgecolor='black', label='G = Goal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               facecolor='white', edgecolor='#ccc', labelcolor='black', fontsize=10,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Learned Policy Maps — Q-Learning vs SARSA",
                 fontsize=14, fontweight='bold', color='black', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved policy map → {save_path}")


# ─────────────────────────────────────────────
# Fig 2: Optimal Path Map  (style_path.png)
# White background, no gradient, path overlay only
# ─────────────────────────────────────────────

def _draw_path_grid(ax, agent, env, title, path_color):
    """Draw one path-grid panel: plain white grid, cliff shaded,
       softmax probability labels per cell, path overlaid."""
    rows, cols = env.get_grid_shape()
    ax.set_facecolor('white')
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    # ── Cliff shading ──
    for r in range(rows):
        for c in range(cols):
            if (r, c) in env.cliff:
                ax.add_patch(plt.Rectangle((c, r), 1, 1,
                             color='#AED6F1', zorder=1))

    # ── Grid lines ──
    for x in range(cols + 1):
        ax.axvline(x, color='black', linewidth=1.0, zorder=2)
    for y in range(rows + 1):
        ax.axhline(y, color='black', linewidth=1.0, zorder=2)

    # ── Per-cell: softmax probability labels + best-action arrow ──
    # Softmax of Q-values gives relative probability of each direction
    # Label positions (dx, dy from cell center): up, down, left, right
    PROB_OFFSET = {
        0: ( 0.0, -0.28),   # up
        1: ( 0.0,  0.28),   # down
        2: (-0.30,  0.0),   # left
        3: ( 0.30,  0.0),   # right
    }

    for r in range(rows):
        for c in range(cols):
            pos = (r, c)
            cx, cy = c + 0.5, r + 0.5

            if pos == env.start:
                ax.text(cx, cy, 'S', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='black', zorder=5)
                continue
            if pos == env.goal:
                ax.text(cx, cy, 'G', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='black', zorder=5)
                continue
            if pos in env.cliff:
                if c == (env.cols // 2):
                    ax.text(cx, cy, 'Cliff', ha='center', va='center',
                            fontsize=9, color='#1A5276', fontweight='bold', zorder=3)
                continue

            state = r * cols + c
            q_row  = agent.Q[state].cpu().float()
            probs  = torch.softmax(q_row, dim=0).numpy()   # shape (4,)
            best_a = int(q_row.argmax().item())

            # ── Probability labels (tiny text at 4 compass positions) ──
            prob_color = ['#888888', '#888888', '#888888', '#888888']
            prob_color[best_a] = '#C62828'   # highlight greedy action probability in red
            for a in range(4):
                dx, dy = PROB_OFFSET[a]
                ax.text(cx + dx, cy + dy, f'{probs[a]:.2f}',
                        ha='center', va='center', fontsize=5.5,
                        color=prob_color[a], zorder=4)

            # ── Best-action arrow (prominent, black) ──
            u = ACTION_DC[best_a]
            v = -ACTION_DR[best_a]
            ax.annotate("",
                xy    =(cx + u * 0.22, cy - v * 0.22),
                xytext=(cx - u * 0.22, cy + v * 0.22),
                arrowprops=dict(arrowstyle="-|>", color='black',
                                lw=1.8, mutation_scale=16),
                zorder=5
            )

    # ── Optimal path overlay ──
    opt_path = get_optimal_path(agent, env)
    if opt_path is not None:
        px = [p[1] + 0.5 for p in opt_path]
        py = [p[0] + 0.5 for p in opt_path]
        ax.plot(px, py, color=path_color, linewidth=2.8, zorder=6,
                marker='o', markersize=6,
                markerfacecolor=path_color, markeredgecolor='black',
                markeredgewidth=0.8, alpha=0.85)
        ax.plot(px[0],  py[0],  's', color='limegreen',
                markersize=11, zorder=7, markeredgecolor='black', markeredgewidth=1.2)
        ax.plot(px[-1], py[-1], '*', color='gold',
                markersize=16, zorder=7, markeredgecolor='black', markeredgewidth=0.8)
        print(f"    [{title}] path: {len(opt_path)} steps")
    else:
        ax.text(cols / 2, rows / 2, 'No path found', ha='center', va='center',
                fontsize=12, color='red')
        print(f"    [{title}] WARNING: path did not reach Goal")

    # ── Axes cosmetics ──
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold', color='black', pad=8)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)


def plot_optimal_paths(q_agent, s_agent, env, save_path):
    """Save style_path.png: 1x2 subplots, white grid with path overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    _draw_path_grid(axes[0], q_agent, env,
                    "Q-Learning  (Off-policy)", path_color='#E53935')   # red path
    _draw_path_grid(axes[1], s_agent, env,
                    "SARSA  (On-policy)",       path_color='#1565C0')   # blue path

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(color='#AED6F1', label='Cliff area'),
        plt.Line2D([0], [0], color='#E53935', lw=2.5, marker='o', markersize=6,
                   label='Optimal Path (Q-Learning)'),
        plt.Line2D([0], [0], color='#1565C0', lw=2.5, marker='o', markersize=6,
                   label='Optimal Path (SARSA)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='limegreen',
                   markersize=10, markeredgecolor='black', label='S = Start'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=14, markeredgecolor='black', label='G = Goal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               facecolor='white', edgecolor='#ccc', labelcolor='black', fontsize=10,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Optimal Paths — Q-Learning vs SARSA  (Cliff Walking 4x12)",
                 fontsize=14, fontweight='bold', color='black', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved path map → {save_path}")

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
    print("[Plotting] style_policy.png — policy direction maps...")
    plot_policy_arrows(q_agent, s_agent, env, STYLE_POLICY_PATH)
    print("[Plotting] style_path.png — optimal path maps...")
    plot_optimal_paths(q_agent, s_agent, env, STYLE_PATH_PATH)

    print("\nAll done! Results saved to:")
    print(f"   - {GRAPH_PATH}")
    print(f"   - {STYLE_POLICY_PATH}")
    print(f"   - {STYLE_PATH_PATH}")

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
