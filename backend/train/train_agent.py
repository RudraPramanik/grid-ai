import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../energy_env')))

from energy_env.energy_env import EnergyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd

# Path to the dataset
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/energy.csv'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/ppo_energy'))
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/'))

def main():
    # Create environment directly
    env = EnergyEnv(csv_path=DATA_PATH)
    env = Monitor(env, LOG_PATH)

    # Create RL agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main() 