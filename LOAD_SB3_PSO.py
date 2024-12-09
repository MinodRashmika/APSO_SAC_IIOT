import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from PSO_Env_RL_Final import PSOEnv 

# Initialize the custom PSO environment
env = PSOEnv()

models_dir = "../models/DDPG"
model_path = f"{models_dir}/50000.zip"

# Load the trained model
model = DDPG.load(model_path, env=env)

# Define action noise (OU Noise) again when loading
n_actions = env.action_space.shape[0]
ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions), theta=0.15, dt=1e-2)

episodes = 10


for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.render()

env.close()
