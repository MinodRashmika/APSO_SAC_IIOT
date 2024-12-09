import numpy as np
import pyswarms as ps
import time
import matplotlib.pyplot as plt

# Set random seed for consistency
np.random.seed(21)

# Number of servers and devices
n_servers = 20
n_devices = 250

# Simulation parameters (consistent)
network_speed = np.random.uniform(60, 900, n_devices)  # Mbps
server_cost = np.random.uniform(0.02, 0.06, n_servers)  # $/second
server_speed = np.random.uniform(10, 200, n_servers)  # MI/second
server_ram = np.random.uniform(2, 8, n_servers)  # GB
data_size = np.random.uniform(50, 150, n_devices)  # MB
completion_requirement = np.random.uniform(20, 40, n_devices)  # MI
ram_requirement = np.random.uniform(1, 2, n_devices)  # GB

# Weights for the cost and latency components
m = 10  # cost weight
n = 1e-2  # latency weight

# Function to generate consistent workloads and server capacities
def generate_data():
    device_workloads = np.random.uniform(1e6, 1e8, n_devices)
    return device_workloads


def calculate_total_cost(device_workloads, positions, data_size, network_speed, completion_requirement, server_speed, server_cost, server_ram, ram_requirement, m, n):
    total_costs = np.zeros(positions.shape[0])

    for i in range(positions.shape[0]):  # Iterate over particles or server allocations
        particle = positions[i]
        total_cost = 0
        
        for j in range(n_devices):  # For each device
            server_index = int(particle[j])  # The server assigned to the device
            
            transfer_time = data_size[j] / network_speed[j]  # Time to transfer data (seconds)
            processing_time = completion_requirement[j] / server_speed[server_index]  # Time to process (seconds)
            total_time = transfer_time + processing_time
            
            if ram_requirement[j] <= server_ram[server_index]:  # Check if server has enough RAM
                current_cost = server_cost[server_index] * total_time
                total_cost += m * current_cost + n * total_time  # Weighted cost and latency
        
        total_costs[i] = total_cost  # Store the total cost for this configuration

    return total_costs


# PSO objective function
def pso_objective_function(positions, device_workloads):
    return calculate_total_cost(device_workloads, positions, data_size, network_speed, completion_requirement, server_speed, server_cost, server_ram, ram_requirement, m, n)



# Timing and testing PSO
def test_pso(device_workloads, iterations):
    runtimes = []
    best_costs = []
    for _ in range(iterations):
        start_time = time.time()
        optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=n_devices, options={'c1': 2, 'c2': 2, 'w': 0.9}, bounds=(np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices)))
        cost, _ = optimizer.optimize(lambda pos: pso_objective_function(pos, device_workloads), iters=50)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        best_costs.append(cost)  # Collect the best cost of each run
    return best_costs  # Return only best costs

# Test the algorithm with different iteration counts
iterations = 10  # Number of iterations for testing
device_workloads = generate_data()  # Generate device workloads once for testing
best_costs = test_pso(device_workloads, iterations)

# Calculate average best cost
average_best_cost = np.mean(best_costs)

# Plotting the results in 2D
plt.figure(figsize=(10, 6))
plt.plot(range(1, iterations + 1), best_costs, marker='o', linestyle='-', color='b', label='Best Cost')

# Plot the average best cost
plt.axhline(y=average_best_cost, color='r', linestyle='--', label='Average Best Cost')

# Add a label for the average best cost
plt.text(iterations, average_best_cost + 0.5, f'Average Best Cost: \n {average_best_cost:.2f}', color='r', 
         verticalalignment='bottom', horizontalalignment='left', fontsize=10)

plt.xlabel('Run Number')
plt.ylabel('Best Cost')
plt.title('Best Cost in Each PSO Run')
plt.xticks(range(1, iterations + 1))  # Set x-ticks to run numbers
plt.legend()
plt.grid(True)
plt.show()
