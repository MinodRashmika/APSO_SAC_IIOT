import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from PSO_Env_RL_Final import PSOEnv 

def make_env():
    def _init():
        env = PSOEnv()
        env = Monitor(env)  # Optional: Monitor to log statistics
        return env
    return _init
def main():
    # Create directories for models and logs if they don't exist
    models_dir = "../models/TD3"
    logdir = "../logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #env = PSOEnv()


    # Initialize the custom PSO environment
    single_env = PSOEnv()
    check_env(single_env)
    single_env.close()

    # Define the number of parallel environments
    num_envs = 4  # Adjust based on your CPU cores

    # Create the vectorized environment
    env = make_vec_env(
        make_env(),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        # Optional: Set a different seed for each environment
        # seed=42
    )

    # Define action noise (OU Noise)
    n_actions = env.action_space.shape[0] 
    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), 
                                            sigma=0.5 * np.ones(n_actions), 
                                            theta=0.15, 
                                            dt=1e-2)
    
    # Initialize the DDPG model
    model = TD3("MlpPolicy", 
                env, 
                learning_rate=1e-3, 
                action_noise=ou_noise, 
                verbose=1, 
                batch_size=4096, 
                tensorboard_log=logdir
                )

    # Training loop
    TIMESTEPS = 1000
    EPISODES = 200

    for i in range(1, EPISODES + 1):
        model.learn(total_timesteps=TIMESTEPS, 
                    reset_num_timesteps=False, 
                    tb_log_name="TD3")
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    env.close()

if __name__ == '__main__':
    main()