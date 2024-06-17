# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:21:49 2024

@author: nicol
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

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
simulation_time_steps = 100

# Function to calculate cost value for an individual
def calculate_cost(individual, resources):
    cost = np.maximum(individual['resources_trait'] - resources, 0)
    return cost

# Read resources from CSV file
resource_values_df = pd.read_csv('resources.csv', header=None)
resource_values = resource_values_df.values

# Initialize resources_values array for all time steps
resources_values = np.zeros((simulation_time_steps + 1, L, L))
resources_values[0] = resource_values

# Initialize population with random traits
population = []
for _ in range(500):
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)
    resources_trait = random.uniform(8, 10)
    cost = calculate_cost({'resources_trait': resources_trait}, resources_values[0, x, y])
    population.append({'x': x, 'y': y, 'resources_trait': resources_trait, 'cost': cost})

# Function to perform reproduction
def simulation(population):
    num_individuals_over_time = []
    for t in range(simulation_time_steps):
        new_population = []
        num_births = 0
        num_deaths = 0
        num_moves = 0

        # Movement
        for individual in population:
            movement_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if individual['cost'] > 0:
                movement_probability = 0.1
            else:
                movement_probability = 0.01
            if random.random() < movement_probability:
                dx, dy = movement_options[random.randint(0, 3)]
                individual['x'] = (individual['x'] + dx) % L
                individual['y'] = (individual['y'] + dy) % L
                num_moves += 1

        # Death rate calculation and population update
        death_list = []
        for individual in population:
            if individual["cost"] > 4:
                death_list.append(individual)

        for individual in death_list:
            population.remove(individual)
            num_deaths += 1

        for individual in population:
            if random.random() < 0.2:
                offspring = {'x': individual['x'], 'y': individual['y'], 'resources_trait': random.uniform(8, 10)}
                offspring['cost'] = calculate_cost(offspring, resources_values[t, offspring['x'], offspring['y']])
                new_population.append(offspring)
                num_births += 1

        population.extend(new_population)

        print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Number of Individuals = {len(population)}")
        
        # Store number of individuals for plotting later
        num_individuals_over_time.append(len(population))

        population_density = np.zeros((L, L))
        for individual in population:
            population_density[individual['x'], individual['y']] += 1

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(population_density[:, :], aspect='auto', cmap='viridis', extent=[0, L, 0, L])
        plt.colorbar(label='Population Density')

        plt.subplot(1, 2, 2)
        plt.imshow(resources_values[0, :, :], aspect='auto', cmap='viridis', extent=[0, L, 0, L])
        plt.colorbar(label='Resources')
        plt.show()
    
    # Plot number of individuals over time
    plt.figure()
    plt.plot(num_individuals_over_time)
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Individuals')
    plt.title('Number of Individuals Over Time')
    plt.show()

# Perform simulation
simulation(population)
