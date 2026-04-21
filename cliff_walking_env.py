"""
Cliff Walking Environment (4x12 Grid)
======================================
- Start: (3, 0) bottom-left
- Goal:  (3, 11) bottom-right
- Cliff: (3, 1) ~ (3, 10)
- Reward: -1 per step, -100 for cliff (reset to start), episode ends at goal
"""

import numpy as np


class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        # Cliff positions: bottom row between start and goal
        self.cliff = [(3, c) for c in range(1, 11)]

        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = 4
        self.state_space = self.rows * self.cols

        self.state = None
        self.reset()

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start
        return self._state_to_idx(self.state)

    def step(self, action):
        """
        Take an action and return (next_state_idx, reward, done).
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        r, c = self.state

        if action == 0:    # Up
            r = max(r - 1, 0)
        elif action == 1:  # Down
            r = min(r + 1, self.rows - 1)
        elif action == 2:  # Left
            c = max(c - 1, 0)
        elif action == 3:  # Right
            c = min(c + 1, self.cols - 1)

        next_pos = (r, c)

        # Check cliff
        if next_pos in self.cliff:
            self.state = self.start
            return self._state_to_idx(self.start), -100, False

        # Check goal
        if next_pos == self.goal:
            self.state = next_pos
            return self._state_to_idx(next_pos), -1, True

        # Normal move
        self.state = next_pos
        return self._state_to_idx(next_pos), -1, False

    def _state_to_idx(self, pos):
        """Convert (row, col) position to flat state index."""
        return pos[0] * self.cols + pos[1]

    def _idx_to_state(self, idx):
        """Convert flat state index to (row, col) position."""
        return (idx // self.cols, idx % self.cols)

    def get_grid_shape(self):
        return self.rows, self.cols
