import numpy as np
import pyswarms as ps
import time
import matplotlib.pyplot as plt

# Number of servers and devices
n_servers = 3
n_devices = 30

# Static simulation parameters
network_speed = np.array([100] * n_devices)  # Mbps
server_cost = np.array([0.03] * n_servers)  # $/second
server_speed = np.array([100] * n_servers)  # MI/second
server_ram = np.array([4] * n_servers)  # GB
data_size = np.array([100] * n_devices)  # MB
completion_requirement = np.array([30] * n_devices)  # MI
ram_requirement = np.array([1.5] * n_devices)  # GB

# Weights for the cost and latency components
m = 10  # cost weight
n = 1e-2  # latency weight

# Function to generate static workloads (can be removed if not needed)
def generate_data():
    device_workloads = np.ones(n_devices) * 1e7  # Static workload
    return device_workloads

# PSO objective function
def pso_objective_function(positions, device_workloads):
    total_costs = np.zeros(positions.shape[0])  # Initialize total costs for each particle

    for i in range(positions.shape[0]):  # Iterate over particles
        particle = positions[i]
        total_cost = 0
        for j in range(n_devices):  # For each device
            server_index = int(particle[j])
            transfer_time = data_size[j] / network_speed[j]  # Time to transfer data (seconds)
            processing_time = completion_requirement[j] / server_speed[server_index]  # Time to process (seconds)
            total_time = transfer_time + processing_time
            
            if ram_requirement[j] <= server_ram[server_index]:  # Check if server has enough RAM
                current_cost = server_cost[server_index] * total_time
                total_cost += m * current_cost + n * total_time  # Weighted cost and latency

        total_costs[i] = total_cost  # Store the total cost for this particle

    return total_costs

# Timing and testing PSO
def test_pso(device_workloads, iterations):
    runtimes = []
    best_costs = []
    for _ in range(iterations):
        start_time = time.time()
        optimizer = ps.single.GlobalBestPSO(
            n_particles=150,  # Increased number of particles
            dimensions=n_devices,
            options={
                'c1': 1.5,  # Adjusted cognitive coefficient
                'c2': 1,  # Adjusted social coefficient
                'w': 0.7    # Adjusted inertia weight
            },
            bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices))
        )
        cost, _ = optimizer.optimize(lambda pos: pso_objective_function(pos, device_workloads), iters=100)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        best_costs.append(cost)
    return runtimes, best_costs

# Test the algorithm with a single iteration count
iterations = 3  # Total runs
runtimes, best_costs = test_pso(generate_data(), iterations)

# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(runtimes, best_costs, marker='o', linestyle='-')

plt.xlabel('Runtime (seconds)')
plt.ylabel('Best Cost')
plt.title('PSO Performance: Runtime vs Best Cost for Each Run')
plt.grid(True)
plt.show()
