import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 10  # Grid size
population_size = 100
simulation_time_steps = 100

# Function to initialize population with random traits
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        x = np.random.randint(0, L)  # Generate x coordinate from 0 to L-1
        y = np.random.randint(0, L)  # Generate y coordinate from 0 to L-1
        population.append({'x': x, 'y': y})
    return population

# Function to simulate aridity gradient
def simulate_aridity_gradient():
    aridity_values = np.random.rand(L, L) * 10  # Random aridity values
    return aridity_values

# Function to simulate resources gradient based on aridity and population density
def simulate_resources_gradient(population, aridity_values):
    resources_values = np.zeros((simulation_time_steps + 1, L, L))
    # Generate initial resources distribution based on aridity
    for i in range(L):
        for j in range(L):
            if aridity_values[i, j] > 5:  # High aridity leads to low resources
                resources_values[0, i, j] = np.random.uniform(0, 3)
            else:  # Low aridity leads to high resources
                resources_values[0, i, j] = np.random.uniform(7, 10)

    # Update resources based on population density after each time step
    for t in range(1, simulation_time_steps + 1):
        for i in range(L):
            for j in range(L):
                # Calculate population density at the current location
                population_density = sum(1 for individual in population if individual['x'] == i and individual['y'] == j)
                # Reduce resources based on population density
                resources_values[t, i, j] = max(resources_values[t - 1, i, j] - population_density * 0.1, 0)
    return resources_values


# Function to plot resources distribution over time
def plot_resources_distribution(resources_values):
    for t in range(simulation_time_steps + 1):
        plt.figure(figsize=(6, 6))
        plt.imshow(resources_values[t, :, :], aspect='auto', cmap='viridis', extent=[0, L, 0, L])
        plt.colorbar(label='Resources')
        plt.title(f'Resources Distribution - Time Step {t}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

# Main simulation
population = initialize_population(population_size)
aridity_values = simulate_aridity_gradient()
resources_values = simulate_resources_gradient(population, aridity_values)
plot_resources_distribution(resources_values)
