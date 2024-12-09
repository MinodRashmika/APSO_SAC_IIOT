import numpy as np
import pyswarms as ps
import time
import matplotlib.pyplot as plt
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import Star

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

def calculate_total_cost(device_workloads, positions, data_size, network_speed,
                         completion_requirement, server_speed, server_cost,
                         server_ram, ram_requirement, m, n):
    total_costs = np.zeros(positions.shape[0])

    for i in range(positions.shape[0]):  # Iterate over particles or server allocations
        particle = positions[i]
        total_cost = 0

        for j in range(n_devices):  # For each device
            server_index = int(np.clip(particle[j], 0, n_servers - 1))  # Ensure valid server index

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
    return calculate_total_cost(device_workloads, positions, data_size, network_speed,
                                completion_requirement, server_speed, server_cost,
                                server_ram, ram_requirement, m, n)

# Evolutionary State Estimation (ESE) function
def evolutionary_state_estimation(swarm):
    """
    Estimates the evolutionary factor f of the swarm.
    """
    # Calculate population diversity (normalized)
    position_std = np.mean(np.std(swarm.position, axis=0))
    max_std = np.sqrt(np.sum((swarm.options['bounds'][1] - swarm.options['bounds'][0])**2))
    norm_diversity = position_std / max_std

    # Calculate relative fitness
    fitness = swarm.pbest_cost
    best_fitness = np.min(fitness)
    worst_fitness = np.max(fitness)
    if worst_fitness - best_fitness == 0:
        norm_fitness = 0
    else:
        norm_fitness = (fitness.mean() - best_fitness) / (worst_fitness - best_fitness)

    # Evolutionary factor f combines diversity and relative fitness
    f = norm_fitness

    return f

# Adaptive parameter control based on ESE
def adaptive_parameters(swarm, f, delta):
    """
    Adjusts the inertia weight and acceleration coefficients based on the evolutionary factor f.
    """
    # Adapt inertia weight ω(f) using the sigmoid mapping
    omega = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    omega = np.clip(omega, 0.4, 0.9)
    swarm.options['w'] = omega

    # Retrieve current c1 and c2
    c1 = swarm.options.get('c1', 2.0)
    c2 = swarm.options.get('c2', 2.0)

    # Determine evolutionary state based on f
    if f < 0.2:
        state = 'convergence'
    elif f < 0.4:
        state = 'exploitation'
    elif f < 0.6:
        state = 'exploration'
    else:
        state = 'jumping_out'

    # Adjust c1 and c2 based on the strategies
    if state == 'exploration':
        # Strategy 1: Increase c1, Decrease c2
        c1 = c1 + delta
        c2 = c2 - delta
    elif state == 'exploitation':
        # Strategy 2: Slightly Increase c1, Slightly Decrease c2
        c1 = c1 + 0.5 * delta
        c2 = c2 - 0.5 * delta
    elif state == 'convergence':
        # Strategy 3: Slightly Increase c1, Slightly Increase c2
        c1 = c1 + 0.5 * delta
        c2 = c2 + 0.5 * delta
    elif state == 'jumping_out':
        # Strategy 4: Decrease c1, Increase c2
        c1 = c1 - delta
        c2 = c2 + delta
        # Apply Gaussian perturbation-based ELS
        elitist_learning(swarm)

    # Bound the changes by δ
    c1_change = np.clip(c1 - swarm.options['c1'], -delta, delta)
    c2_change = np.clip(c2 - swarm.options['c2'], -delta, delta)
    c1 = swarm.options['c1'] + c1_change
    c2 = swarm.options['c2'] + c2_change

    # Clamp c1 and c2 to [1.5, 2.5]
    c1 = np.clip(c1, 1.5, 2.5)
    c2 = np.clip(c2, 1.5, 2.5)

    # Ensure sum of c1 and c2 is within [3.0, 4.0]
    c_sum = c1 + c2
    if c_sum > 4.0:
        c1 = c1 / c_sum * 4.0
        c2 = c2 / c_sum * 4.0
    elif c_sum < 3.0:
        c1 = c1 / c_sum * 3.0
        c2 = c2 / c_sum * 3.0

    # Update the swarm's options
    swarm.options['c1'] = c1
    swarm.options['c2'] = c2

# Elitist Learning Strategy (ELS)
def elitist_learning(swarm):
    """
    Applies Gaussian perturbation to the global best position to help escape local optima.
    """
    learning_rate = 0.1  # Time-varying learning rate can be implemented if needed
    perturbation = np.random.normal(0, learning_rate, swarm.dimensions)
    swarm.best_pos += perturbation

# Custom optimizer class to implement APSO
def adaptive_pso(n_particles, dimensions, options, bounds, iters, objective_func):
    # Initialize the swarm
    swarm = Swarm(position=np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions)),
                  velocity=np.zeros((n_particles, dimensions)),
                  options=options)
    swarm.options['bounds'] = bounds  # Store bounds in options for later use

    # Initialize the personal best positions and costs
    swarm.pbest_pos = np.copy(swarm.position)
    swarm.pbest_cost = np.full(n_particles, np.inf)

    # Initialize the global best position and cost
    swarm.best_pos = np.zeros(dimensions)
    swarm.best_cost = np.inf

    delta_min, delta_max = 0.05, 0.1

    for i in range(iters):
        # Compute cost for current positions
        swarm.current_cost = objective_func(swarm.position)

        # Update personal bests
        mask = swarm.current_cost < swarm.pbest_cost
        swarm.pbest_pos[mask] = swarm.position[mask]
        swarm.pbest_cost[mask] = swarm.current_cost[mask]

        # Update global best
        best_cost_arg = np.argmin(swarm.pbest_cost)
        if swarm.pbest_cost[best_cost_arg] < swarm.best_cost:
            swarm.best_cost = swarm.pbest_cost[best_cost_arg]
            swarm.best_pos = swarm.pbest_pos[best_cost_arg].copy()

        # Evolutionary State Estimation
        f = evolutionary_state_estimation(swarm)

        # Adaptive parameter control
        delta = np.random.uniform(delta_min, delta_max)
        adaptive_parameters(swarm, f, delta)

        # Update velocities and positions
        r1 = np.random.rand(n_particles, dimensions)
        r2 = np.random.rand(n_particles, dimensions)
        cognitive = swarm.options['c1'] * r1 * (swarm.pbest_pos - swarm.position)
        social = swarm.options['c2'] * r2 * (swarm.best_pos - swarm.position)
        swarm.velocity = swarm.options['w'] * swarm.velocity + cognitive + social

        # Update positions
        swarm.position += swarm.velocity

        # Apply bounds
        swarm.position = np.clip(swarm.position, bounds[0], bounds[1])

    return swarm.best_cost, swarm.best_pos

# Timing and testing APSO
def test_apso(device_workloads, iterations):
    runtimes = []
    best_costs = []
    for _ in range(iterations):
        start_time = time.time()
        n_particles = 50
        dimensions = n_devices
        options = {'c1': 2.0, 'c2': 2.0, 'w': 0.9}
        bounds = (np.zeros(n_devices), (n_servers - 1) * np.ones(n_devices))

        best_cost, _ = adaptive_pso(n_particles, dimensions, options, bounds, iters=50,
                                    objective_func=lambda pos: pso_objective_function(pos, device_workloads))
        end_time = time.time()
        runtimes.append(end_time - start_time)
        best_costs.append(best_cost)  # Collect the best cost of each run
    return best_costs  # Return only best costs

# Test the APSO algorithm with different iteration counts
iterations = 10  # Number of iterations for testing
device_workloads = generate_data()  # Generate device workloads once for testing
best_costs = test_apso(device_workloads, iterations)

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
plt.title('Best Cost in Each APSO Run')
plt.xticks(range(1, iterations + 1))  # Set x-ticks to run numbers
plt.legend()
plt.grid(True)
plt.show()
