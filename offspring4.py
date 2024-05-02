# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:40:50 2024

@author: nicol
"""
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
simulation_time_steps = 500

# Define world areas
dune_range = (4, 6)  # Dune area
tropics_range = (2, 8)  # Tropics area
ice_range = (0, 2)  # Ice area

# Initialize population with random traits
population = []
for _ in range(500):
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)
    temperature_trait = random.uniform(0, 10)
    aridity_trait = random.uniform(0, 10)
    resources_trait = random.uniform(0, 10)
    cost_trait = random.uniform(0, 10)
    population.append({'x': x, 'y': y, 'temperature_trait': temperature_trait, 'aridity_trait': aridity_trait,
                       'resources_trait': resources_trait, 'cost_trait': cost_trait})

# Simulate aridity gradient
aridity_values = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if j > 5:
            aridity_values[i, j] = 10
        else:
            aridity_values[i, j] = np.random.uniform(0, 5)

# Simulate temperature gradient
temperature_values = np.zeros((simulation_time_steps + 1, L, L))
for t in range(simulation_time_steps + 1):
    for i in range(L):
        for j in range(L):
            if j > 5:
                temperature_values[t, i, j] = np.random.uniform(25, 35)
            else:
                temperature_values[t, i, j] = np.random.uniform(15, 25)

# Simulate resources gradient based on world areas
resources_values = np.zeros((simulation_time_steps + 1, L, L))
for i in range(L):
    for j in range(L):
        if dune_range[0] <= i <= dune_range[1]:  # Dune area
            resources_values[0, i, j] = 0
        elif tropics_range[0] <= i <= tropics_range[1]:  # Tropics area
            resources_values[0, i, j] = 10
        elif ice_range[0] <= i <= ice_range[1]:  # Ice area
            resources_values[0, i, j] = 5

# Function to calculate cost value for an individual
def calculate_cost(individual, temperature, resources):
    cost = abs(individual['temperature_trait'] - temperature) + abs(individual['resources_trait'] - resources)
    return cost

# Function to calculate phenotype-based mating probability between two individuals
def calculate_phenotype_based_mating_probability(individual1, individual2):
    cost_difference = abs(individual1['cost_trait'] - individual2['cost_trait'])
    mating_probability = math.exp(-cost_difference)
    return mating_probability

# Function to calculate spatial distance between two individuals
def calculate_spatial_distance(individual1, individual2):
    distance = math.sqrt((individual1['x'] - individual2['x'])**2 + (individual1['y'] - individual2['y'])**2)
    return distance

# Function to calculate spatial distance-based mating probability
def calculate_spatial_distance_based_mating_probability(individual1, individual2):
    if individual1['x'] == individual2['x'] and individual1['y'] == individual2['y']:
        return 1
    else:
        return 0

# Function to perform reproduction
def birth(population):
    for individual in population:
        # Calculate phenotype-based mating probability
        mating_probabilities = []
        for other_individual in population:
            if other_individual != individual:
                mating_probabilities.append(calculate_phenotype_based_mating_probability(individual, other_individual))
        sum_mating_probabilities = sum(mating_probabilities)
        normalized_mating_probabilities = [prob / sum_mating_probabilities for prob in mating_probabilities]

        # Calculate spatial distance-based mating probability
        spatial_distances = []
        for other_individual in population:
            if other_individual != individual:
                spatial_distances.append(calculate_spatial_distance(individual, other_individual))
        sum_spatial_distances = sum(spatial_distances)
        normalized_spatial_distances = [dist / sum_spatial_distances for dist in spatial_distances]

        # Calculate number of suitable mating partners locally available to individual i
        n_p = sum([prob * dist for prob, dist in zip(normalized_mating_probabilities, normalized_spatial_distances)])

        # Calculate birth rate considering rarity cost
        rarity_cost = b / (1 + c / n_p)

        # Check if individual reproduces
        if random.random() < rarity_cost:
            # Offspring inherits parent's location
            offspring = {'x': individual['x'], 'y': individual['y'],
                         'temperature_trait': random.uniform(0, 10),
                         'resources_trait': random.uniform(0, 10),
                         'cost_trait': random.uniform(0, 10)}
            population.append(offspring)


# Inside the movement loop
num_individuals_over_time = []
for t in range(simulation_time_steps):
    new_population = []
    num_births = 0
    num_deaths = 0
    num_moves = 0

    # Birth process
    previous_population = len(population)
    birth(population)
    num_births = len(population) - previous_population

    # Death rate calculation and population update
    num_deaths = len(population) - sum(1 for individual in population if random.random() > (
            sum(1 for other_individual in population if other_individual != individual) /
            (a * (individual['x'] - L / 2) + L / 2) * K0 * math.exp(-0.5 * (
                (individual['cost_trait'] - (a * (individual['x'] - L / 2) + L / 2)) / sigma_k) ** 2)))
    population = population[:len(population) - num_deaths]


    # Movement
    for individual in population:
        scaled_x = int(individual['x'])
        scaled_y = int(individual['y'])
        cost = abs(individual['temperature_trait'] - temperature_values[0, scaled_x, scaled_y]) \
               + max(0, 5 - resources_values[t, scaled_x, scaled_y])
        individual['u'] = cost
    
        # Check if there are more than 3 individuals in the same location
        num_individuals_same_location = sum(1 for other_individual in population if other_individual['x'] == individual['x'] and other_individual['y'] == individual['y'])
        
        if cost > 25 or num_individuals_same_location > 10: 
            movement_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if individual['resources_trait'] > 7:  # Animals doing well have less chance of moving
                movement_probabilities = [0.1, 0.2, 0.3, 0.4]  # Adjust probabilities as needed
            else:
                movement_probabilities = [0.25, 0.25, 0.25, 0.25]
            dx, dy = random.choices(movement_options, weights=movement_probabilities)[0]
            new_x = max(0, min(individual['x'] + dx, L - 1))
            new_y = max(0, min(individual['y'] + dy, L - 1))
            individual['x'] = new_x
            individual['y'] = new_y
            num_moves += 1


    num_individuals_over_time.append(len(population))

    print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Number of Individuals = {len(population)}")

    # Plot population distribution and resources distribution
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    x = [individual['x'] for individual in population]
    y = [individual['y'] for individual in population]
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.title(f'Population Distribution - Time Step {t}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(resources_values[0, :, :], aspect='auto', cmap='viridis', extent=[0, L, 0, L])
    plt.colorbar(label='Resources')
    plt.title(f'Resources Distribution - Initial')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot number of individuals over time
plt.figure()
plt.plot(range(1, simulation_time_steps + 1), num_individuals_over_time, marker='o', linestyle='-', color='b')
plt.title('Number of Individuals Over Time')
plt.xlabel('Time Step')
plt.ylabel('Number of Individuals')
plt.grid(True)
plt.show()
