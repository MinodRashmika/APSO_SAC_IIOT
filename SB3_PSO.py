import numpy as np
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from PSO_Env_RL_Final import PSOEnv

# Initialize the custom PSO environment
env = PSOEnv()

# Check if the environment follows the Gym API
check_env(env)

# Use DDPG or TD3 (both can handle continuous action spaces)
model = TD3("MlpPolicy", env, verbose=1)

# Train the RL agent
model.learn(total_timesteps=40)

# Save the trained model
model.save("pso_ddpg_td3_model")

episodes = 2

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()