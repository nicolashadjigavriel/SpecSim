import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define parameters
L = 10  # Grid size
K0 = 500
sigma_k = 0.3
a = 0.95
sigma_c = 0.9
sigma_s = 0.19
b = 1
mu_a = 0.005
sigma_a = 0.05
sigma_p = 0.2
c = 10
mu_r = 0.001
m = 5
sigma_m_tilde = 0.27
simulation_time_steps = 10

# Define world areas
dune_range = (4, 6)  # Dune area
tropics_range = (2, 8)  # Tropics area
ice_range = (0, 2)  # Ice area

# Function to initialize population with random traits
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        x = random.uniform(0, L)  # Generate x coordinate from 0 to L
        y = random.uniform(0, L)  # Generate y coordinate from 0 to L
        temperature_trait = random.uniform(0, 10)  # Random temperature trait
        aridity_trait = random.uniform(0, 10)  # Random aridity trait
        resources_trait = random.uniform(0, 10)  # Random resources trait
        cost_trait = random.uniform(0, 10)  # Random cost trait
        population.append({'x': x, 'y': y, 'temperature_trait': temperature_trait, 'aridity_trait': aridity_trait, 'resources_trait': resources_trait, 'cost_trait': cost_trait})
    return population

# Function to simulate temperature gradient
def simulate_temperature_gradient():
    temperature_values = np.zeros((simulation_time_steps + 1, L, L))
    for t in range(simulation_time_steps + 1):
        temperature_values[t, :, :] = np.random.rand(L, L)  # Random temperature trait
    return temperature_values

# Function to simulate resources gradient
def simulate_resources_gradient():
    resources_values = np.zeros((simulation_time_steps + 1, L, L))
    for t in range(simulation_time_steps + 1):
        resources_values[t, :, :] = np.random.rand(L, L)  # Random resources trait
    return resources_values

# Function to calculate aridity
def calculate_aridity(precipitation):
    return 1 - precipitation

# Function to calculate precipitation based on world areas
def calculate_precipitation(x):
    if dune_range[0] < x < dune_range[1]:
        return 1
    elif tropics_range[0] < x < tropics_range[1]:
        return 7
    elif ice_range[0] < x < ice_range[1]:
        return 8
    else:
        return 5  # Default precipitation value for other areas

# Function to calculate carrying capacity for an individual
def calculate_carrying_capacity(individual):
    u0 = a * (individual['x'] - L / 2) + L / 2
    return K0 * math.exp(-0.5 * ((individual['cost_trait'] - u0) / sigma_k) ** 2)

# Function to calculate death rate for an individual
def calculate_death_rate(individual, population):
    effective_density = 0
    for other_individual in population:
        if other_individual != individual:
            delta_cost = abs(individual['cost_trait'] - other_individual['cost_trait'])
            d = math.sqrt((individual['x'] - other_individual['x']) ** 2 + (individual['y'] - other_individual['y']) ** 2)
            effective_density += 1 / (2 * math.pi * sigma_s ** 2 * sigma_c) * math.exp(-0.5 * (delta_cost / sigma_c) ** 2) * math.exp(-0.5 * (d / sigma_s) ** 2)
    return effective_density / calculate_carrying_capacity(individual)

# Function to calculate cost value for an individual
def calculate_cost(individual, temperature, aridity, resources):
    cost = abs(individual['temperature_trait'] - temperature) + aridity + abs(individual['resources_trait'] - resources)
    return cost

# Function to perform reproduction
def birth(population):
    new_population = []
    for individual in population:
        birth_rate = b
        if random.random() < mu_a:  # Mutation
            individual['temperature_trait'] = random.uniform(0, 10)  # Mutation in temperature trait
            individual['aridity_trait'] = random.uniform(0, 10)  # Mutation in aridity trait
            individual['resources_trait'] = random.uniform(0, 10)  # Mutation in resources trait
            individual['cost_trait'] = random.uniform(0, 10)  # Mutation in cost trait
        if random.random() < birth_rate:
            offspring = {'x': individual['x'], 'y': individual['y'], 'temperature_trait': individual['temperature_trait'], 'aridity_trait': individual['aridity_trait'], 'resources_trait': individual['resources_trait'], 'cost_trait': individual['cost_trait']}
            offspring['temperature_trait'] = random.uniform(0, 10)  # Mutation in temperature trait
            offspring['aridity_trait'] = random.uniform(0, 10)  # Mutation in aridity trait
            offspring['resources_trait'] = random.uniform(0, 10)  # Mutation in resources trait
            offspring['cost_trait'] = random.uniform(0, 10)  # Mutation in cost trait
            new_population.append(offspring)
    population.extend(new_population)

# Function to perform death
def death(population):
    population[:] = [individual for individual in population if random.random() > calculate_death_rate(individual, population)]

# Function to perform movement
def movement(population, temperature_values, resources_values):
    for individual in population:
        scaled_x = int(individual['x'])  # Scale x coordinate
        scaled_y = int(individual['y'])  # Scale y coordinate
        cost = calculate_cost(individual, temperature_values[0, scaled_x, scaled_y], calculate_aridity(calculate_precipitation(individual['x'])), resources_values[0, scaled_x, scaled_y])
        individual['u'] = cost  # Update cost value (previously 'temperature_trait')
        if cost < 0.5:  # Check if individual can survive
            movement_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements (adjacent cells)
            dx, dy = random.choice(movement_options)
            new_x = max(0, min(individual['x'] + dx, L - 1))  # Boundaries
            new_y = max(0, min(individual['y'] + dy, L - 1))  # Boundaries
            individual['x'] = new_x
            individual['y'] = new_y

# Function to redistribute individuals if more than 10 are present in a grid cell
def redistribute_individuals(population):
    population_grid = [[[] for _ in range(L)] for _ in range(L)]
    for individual in population:
        grid_x = int(individual['x'])
        grid_y = int(individual['y'])
        population_grid[grid_x][grid_y].append(individual)

    for x in range(L):
        for y in range(L):
            if len(population_grid[x][y]) > 10:
                excess_individuals = population_grid[x][y][10:]
                population_grid[x][y] = population_grid[x][y][:10]
                for excess_individual in excess_individuals:
                    dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # Move to an adjacent cell
                    new_x = max(0, min(x + dx, L - 1))
                    new_y = max(0, min(y + dy, L - 1))
                    population_grid[new_x][new_y].append(excess_individual)
                    excess_individual['x'] = new_x
                    excess_individual['y'] = new_y

    # Update population list
    population.clear()
    for x in range(L):
        for y in range(L):
            population.extend(population_grid[x][y])

# Function to simulate population dynamics
def simulate(population, simulation_time_steps, temperature_values, resources_values):
    for t in range(simulation_time_steps):
        birth(population)
        death(population)
        movement(population, temperature_values, resources_values)
        redistribute_individuals(population)
        yield population

# Main simulation
population_size = 10000
population = initialize_population(population_size)
temperature_values = simulate_temperature_gradient()
resources_values = simulate_resources_gradient()

num_individuals_over_time = []
resources_distribution_over_time = []

for t, population_snapshot in enumerate(simulate(population, simulation_time_steps, temperature_values, resources_values)):
    num_individuals_over_time.append(len(population_snapshot))
    resources_distribution_over_time.append(resources_values[t, :, :])

    plt.figure(figsize=(8, 8))
    x = [individual['x'] for individual in population_snapshot]
    y = [individual['y'] for individual in population_snapshot]
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.title(f'Population Distribution - Time Step {t}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.grid(True)
    plt.show()
    

    # Plot resources distribution for each time step
    plt.figure(figsize=(8, 6))
    plt.imshow(resources_values[t, :, :], aspect='auto', cmap='viridis', extent=[0, L, 0, L])
    plt.colorbar(label='Resources')
    plt.title(f'Resources Distribution - Time Step {t}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

plt.figure()
plt.plot(range(simulation_time_steps), num_individuals_over_time, color='green')
plt.title('Number of Individuals Over Time')
plt.xlabel('Time Step')
plt.ylabel('Number of Individuals')
plt.grid(True)
plt.show()
