"""
RL Agents: Q-Learning (Off-policy) & SARSA (On-policy)
=======================================================
Q-table is stored as a PyTorch tensor and computed on GPU (CUDA) if available,
falling back to CPU gracefully.
"""

import numpy as np
import torch


class BaseAgent:
    """Base class with shared epsilon-greedy logic and Q-table (PyTorch tensor)."""

    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Use CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Using device: {self.device}")

        # Q-table: shape (state_space, action_space), initialized to zeros on device
        self.Q = torch.zeros(state_space, action_space,
                             dtype=torch.float64, device=self.device)

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            # Move to CPU for numpy compatibility if needed
            q_vals = self.Q[state].cpu().numpy()
            return int(np.argmax(q_vals))

    def get_q(self, state, action):
        return self.Q[state, action].item()

    def set_q(self, state, action, value):
        self.Q[state, action] = value

    def get_best_action(self, state):
        return int(torch.argmax(self.Q[state]).item())


class QLearningAgent(BaseAgent):
    """
    Q-Learning (Off-policy TD control)
    Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            max_next_q = torch.max(self.Q[next_state]).item()
            target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (target - current_q)
        self.set_q(state, action, new_q)


class SarsaAgent(BaseAgent):
    """
    SARSA (On-policy TD control)
    Update rule: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    where a' is the ACTUAL next action chosen by epsilon-greedy.
    """

    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q = self.get_q(next_state, next_action)
            target = reward + self.gamma * next_q
        new_q = current_q + self.alpha * (target - current_q)
        self.set_q(state, action, new_q)
