import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class EnergyEnv(gym.Env):
    """
    Custom Environment for Energy Usage Optimization
    State: [current usage, hour of day]
    Action: [0: do nothing, 1: reduce usage]
    Reward: negative of energy cost
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, csv_path='backend/data/energy.csv'):
        super(EnergyEnv, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.current_step = 0
        self.max_steps = len(self.data)
        # Example: state = [usage, hour]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 23]), dtype=np.float32)
        # Example: 0 = do nothing, 1 = reduce usage
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        state = self._get_state()
        return state, {}

    def step(self, action):
        usage = self.data.iloc[self.current_step]['usage']
        hour = self.data.iloc[self.current_step]['hour']
        # Simple action: reduce usage by 10% if action==1
        if action == 1:
            usage *= 0.9
        # Reward: negative cost (e.g., usage * price)
        price = self.data.iloc[self.current_step].get('price', 1.0)
        reward = -usage * price
        self.current_step += 1
        done = self.current_step >= self.max_steps
        state = self._get_state()
        info = {}
        return state, reward, done, False, info

    def _get_state(self):
        if self.current_step >= self.max_steps:
            return np.array([0, 0], dtype=np.float32)
        usage = self.data.iloc[self.current_step]['usage']
        hour = self.data.iloc[self.current_step]['hour']
        return np.array([usage, hour], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")

# Example usage:
# env = EnergyEnv('backend/data/energy.csv')
# obs, _ = env.reset()
# print(obs) 