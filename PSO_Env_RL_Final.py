import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt  # Import for plotting

class PSOEnv(gym.Env):
    """Custom Environment for PSO optimization in mobile edge computing."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action space: c1 and c2 parameters for PSO, normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Define observation space: c1, c2, and last cost achieved
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initialize environment parameters
        self.n_servers = 20
        self.n_devices = 250

        # Set random seed for consistency
        np.random.seed(21)

        # Simulation parameters
        self.network_speed = np.random.uniform(60, 900, self.n_devices)  # Mbps
        self.server_cost = np.random.uniform(0.02, 0.06, self.n_servers)  # $/second
        self.server_speed = np.random.uniform(10, 200, self.n_servers)  # MI/second
        self.server_ram = np.random.uniform(2, 8, self.n_servers)  # GB
        self.data_size = np.random.uniform(50, 150, self.n_devices)  # MB
        self.completion_requirement = np.random.uniform(20, 40, self.n_devices)  # MI
        self.ram_requirement = np.random.uniform(1, 2, self.n_devices)  # GB
        self.device_workloads = np.random.uniform(1e6, 1e8, self.n_devices)

        # Weights for cost and latency components
        self.m = 10  # cost weight
        self.n = 1e-2  # latency weight

        # PSO bounds
        self.bounds = (np.zeros(self.n_devices), (self.n_servers - 1) * np.ones(self.n_devices))

        # PSO parameters
        self.n_particles = 50  # Reduced for faster computation
        self.iters = 50        # Reduced for faster computation
        self.w = 0.9           # Inertia weight

        # Episode management
        self.current_step = 0
        self.max_steps = 1  # Each episode runs for one step

        # Initial state
        self.state = np.zeros(3, dtype=np.float32)

        # Variables to store optimization results
        self.best_cost = None
        self.best_position = None
        self.cost_history = []
        self.pos_history = []

        # List to store the best cost for each episode
        self.episode_best_costs = []
        self.current_episode_best_cost = None  # Best cost for the current episode

        self.c1c2 = True #flag to print c1 and c2 values once on render

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.best_cost = None
        self.best_position = None
        self.cost_history = []
        self.pos_history = []
        self.current_episode_best_cost = None  # Reset best cost for this episode
        info = {}
        return self.state, info

    def step(self, action):
        # Scale action from [-1, 1] to [0.0001, 2.0000]
        c1 = (action[0] + 1.0) * (2.0 - 0.0001) / 2 + 0.0001
        c2 = (action[1] + 1.0) * (2.0 - 0.0001) / 2 + 0.0001

        # Quantize c1 and c2 to four decimal places
        c1 = np.round(c1, 4)
        c2 = np.round(c2, 4)

        # Ensure c1 and c2 are within bounds after rounding
        c1 = np.clip(c1, 0.0001, 2.0000)
        c2 = np.clip(c2, 0.0001, 2.0000)

        # Run PSO with the given c1 and c2
        options = {'c1': c1, 'c2': c2, 'w': self.w}
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles,
                                            dimensions=self.n_devices,
                                            options=options,
                                            bounds=self.bounds)
        cost, pos = optimizer.optimize(self.pso_objective_function, iters=self.iters, verbose=False)

        # Store optimization results
        self.best_cost = cost
        self.best_position = pos
        self.cost_history = optimizer.cost_history
        self.pos_history = optimizer.pos_history  # Positions of particles at each iteration

        # Save the best c1 and c2 values
        self.best_c1 = c1
        self.best_c2 = c2

        # Store the best cost for this episode
        if self.best_cost is not None:
            self.current_episode_best_cost = self.best_cost

        # Update state
        self.state = np.array([c1, c2, cost], dtype=np.float32)

        # Compute reward (negative of cost for minimization)
        reward = -cost / 100 # Scale reward for numerical stability

        # Episode termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if terminated and self.current_episode_best_cost is not None:
            self.episode_best_costs.append(self.current_episode_best_cost)  # Store the best cost for the episode
        truncated = False  # Not using truncation

 
        info = {}
        return self.state, reward, terminated, truncated, info

    def pso_objective_function(self, positions):
        total_costs = np.zeros(positions.shape[0])

        for i in range(positions.shape[0]):  # Iterate over particles
            particle = positions[i]
            total_cost = 0

            for j in range(self.n_devices):  # For each device
                server_index = int(particle[j])  # Assigned server

                transfer_time = self.data_size[j] / self.network_speed[j]  # Transfer time
                processing_time = self.completion_requirement[j] / self.server_speed[server_index]  # Processing time
                total_time = transfer_time + processing_time

                # Check RAM constraints
                if self.ram_requirement[j] <= self.server_ram[server_index]:
                    current_cost = self.server_cost[server_index] * total_time
                    total_cost += self.m * current_cost + self.n * total_time  # Weighted sum

            total_costs[i] = total_cost  # Store total cost

        return total_costs

    def render(self):
        # Check if there are episode results to display
        if self.episode_best_costs:
            print(f"Best c1: {self.best_c1}, Best c2: {self.best_c2}")
            # Calculate the overall average best cost
            average_best_cost = np.mean(self.episode_best_costs)
            print(f"Average Best Cost from PSO: {average_best_cost}")

            # Plotting the results in 2D
            plt.figure(figsize=(10, 6))

            # Plot the best cost for each episode
            episodes = range(1, len(self.episode_best_costs) + 1)
            plt.plot(episodes, self.episode_best_costs, marker='o', linestyle='-', color='b', label='Best Cost per Episode')

            # Plot the overall average best cost
            plt.axhline(y=average_best_cost, color='r', linestyle='--', label=f'Average Best Cost: {average_best_cost:.2f}')

            # Add a label for the average best cost line
            plt.text(len(self.episode_best_costs), average_best_cost + 0.5, f'Avg Best Cost: {average_best_cost:.2f}', 
                    color='r', verticalalignment='bottom', horizontalalignment='left', fontsize=10)

            # Set plot labels, title, and grid
            plt.xlabel('Episode Number')
            plt.ylabel('Best Cost')
            plt.title('Best Cost in Each Episode')
            plt.xticks(episodes)  # Set x-ticks to episode numbers
            plt.legend()
            plt.grid(True)

            # Show the plot
            plt.show()
        else:
            print("No episode results available to display.")


