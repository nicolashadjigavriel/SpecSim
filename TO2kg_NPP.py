# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:27:01 2024

@author: nicol
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Define parameters
simulation_time_steps = 1

# Given constants
#T = 20  # Temperature in Celsius
pO2 = 21.3  # Oxygen partial pressure in kPa
alpha = 1  # Exponent for oxygen in metabolic rate equation
beta = 0.06  # Temperature coefficient
gamma = 2  # Steepness of the logistic curve (example value)
theta = 1  # Threshold ratio where survival probability is 0.5 (example value)
I_rate = 0.03  # Intake rate in kg/day

# Define parameters for sheep (or any other species for testing)
BW_sheep = 70  # Example body weight of a sheep in kg
M = BW_sheep

# Function to calculate metabolic rate R_d
def R_d(M, T, pO2, alpha, beta):
    return 1.5 * (M**(2/3)) * (pO2**alpha) * np.exp(beta * T)

# Function to calculate daily intake I_d
def daily_intake(I_rate, M):
    return I_rate * M

# Function to calculate carrying capacity K
def K(NPP, I_d):
    return NPP / I_d


def survival_probability(K, R_d):
    return 1 / (1 + (R_d / K))

# Read land from CSV file
land_values_lowres_df = pd.read_csv('land_cru.csv', header=None)
land_values_lowres = land_values_lowres_df.values
land_transpose = np.matrix.transpose(land_values_lowres)
land = land_transpose

# Read additional resource data from CSV file
resource_values_lowres_df = pd.read_csv('NPP.csv', header=None)
resource_values_lowres = resource_values_lowres_df.values
resource_transpose = np.matrix.transpose(resource_values_lowres)
resource_scaled = 10 * np.divide(resource_transpose, np.nanmax(resource_transpose))
resource_scaled = np.nan_to_num(resource_scaled, nan=0)
NPP = resource_scaled

# Read latitude and longitude data
lat_df = pd.read_csv('lat_cru.csv', header=None)
lat = lat_df.values
lat = np.squeeze(lat)

lon_df = pd.read_csv('lon_cru.csv', header=None)
lon = lon_df.values
lon = np.squeeze(lon)

# Define the spatial grid
grid_size_lon, grid_size_lat = land.shape



# Load temperature data from CSV
temperature_data = pd.read_csv('tmp_avg.csv', header=None)

# Check the shape of the loaded data 
print(f"Original temperature data shape: {temperature_data.shape}")
temperature_values = temperature_data.values
# Convert the DataFrame to a NumPy array
#temperature_gradient = temperature_data.values
temperature_gradient = np.matrix.transpose(temperature_values)
# Assign the transposed matrix to T (temperature data)
T = np.nan_to_num(temperature_gradient, nan=999)

# Check the final transposed shape (should be (720, 360))
print(f"Final transposed temperature data shape: {T.shape}")

# The rest of your code that uses the variable T can go here


oxygen_gradient = np.linspace(21, 19, grid_size_lat)

# Combine gradients into a 2D grid
# temperature_grid = temperature_gradient
# oxygen_grid = np.tile(oxygen_gradient, (grid_size_lon, 1)).T

# # Verify the corrected grid dimensions
# print(f"Land grid size: {land.shape}")
# print(f"Resource grid size: {resource_scaled.shape}")
# print(f"Temperature grid size: {temperature_grid.shape}")
# print(f"Oxygen grid size: {oxygen_grid.shape}")

population = []
for _ in range(2000):
    while True:
        x = np.random.randint(0, grid_size_lon)
        y = np.random.randint(0, grid_size_lat)
        if land[x, y] == 1 and NPP[x, y] > 0:  # Ensure resources exist in the land area
            break
    biomass = M
    local_T = T[x,y]
    R_d_value = R_d(M, local_T, pO2, alpha, beta)
    print(R_d_value)
    local_NPP = NPP[x, y]
    K_value = K(local_NPP, daily_intake(I_rate, M))
    survival_probability_value = survival_probability(K_value, R_d_value)
    
    individual = {
        'x': x,
        'y': y,
        'biomass': biomass,
        'survival_probability': survival_probability_value,
        'R_d': R_d_value
    }
    population.append(individual)

# Simulation function with boundary checks
# def simulation(population):
num_individuals_over_time = []

# Movement
for t in range(simulation_time_steps):
    new_population = []
    num_births = 0
    num_deaths = 0
    num_moves = 0

    for individual in population:
        # Recalculate survival probability if the individual moves
        if 'survival_probability' not in individual:
            print(f"Missing 'survival_probability' in individual: {individual}")
            individual['survival_probability'] = 0.5  # Assign a default or calculated value
        
        # Decide movement probability based on 'survival_probability'
        if individual['survival_probability'] > 0.75:
            movement_probability = 0.25
        elif individual['survival_probability'] > 0.5:
            movement_probability = 0.75
        else:
            movement_probability = 0.25
        
        # Randomly decide whether the individual moves based on movement_probability
        if random.random() < movement_probability:
            movement_options = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            dx, dy = movement_options[random.randint(0, 3)]
            
            # Test the new position
            test_move_x = individual['x'] + dx
            test_move_y = individual['y'] + dy
            
            if 0 <= test_move_x < grid_size_lon and 0 <= test_move_y < grid_size_lat:
                if land[test_move_x, test_move_y] == 1 and resource_scaled[test_move_x, test_move_y] > 0:
                    individual['x'] = test_move_x
                    individual['y'] = test_move_y
                    num_moves += 1

                    # Recalculate survival probability after movement
                    individual['R_d'] = R_d(individual['biomass'], T[individual['x'],individual['y']], pO2, alpha, beta)
                    K_value = K(NPP[individual['x'], individual['y']], daily_intake(I_rate, individual['biomass']))
                    individual['survival_probability'] = survival_probability(K_value, individual['R_d'])

    # Calculate population density in terms of biomass
    population_density = np.zeros((grid_size_lon, grid_size_lat))
    for individual in population:
        if 0 <= individual['x'] < grid_size_lon and 0 <= individual['y'] < grid_size_lat:
            population_density[individual['x'], individual['y']] += individual['biomass']
            
    # Adjust resources based on population density and consumption rate (phi)
    adjusted_resources = np.copy(resource_scaled)

    # Death rate calculation and population update
    death_list = []
    for individual in population:
        if random.random() > individual["survival_probability"]:
            death_list.append(individual)

    for individual in death_list:
        population.remove(individual)
        num_deaths += 1

    # Births based on existing population
    for individual in population:
        if random.random() < 0.2:
            offspring = {
                'x': individual['x'], 
                'y': individual['y'], 
                'biomass': random.uniform(60, 80),  # Offspring biomass
            }
            if 0 <= offspring['x'] < grid_size_lon and 0 <= offspring['y'] < grid_size_lat:
                # Calculate survival probability for the offspring
                offspring['R_d'] = R_d(offspring['biomass'], T, pO2, alpha, beta)
                K_value = K(NPP[offspring['x'], offspring['y']], daily_intake(I_rate, offspring['biomass']))
                offspring['survival_probability'] = survival_probability(K_value, offspring['R_d'])
                new_population.append(offspring)
                num_births += 1

    population.extend(new_population)

    # Corrected print statement
    print(f"Time Step {t + 1}: Births = {num_births}, Deaths = {num_deaths}, Moves = {num_moves}, Biomass of Population = {sum(ind['biomass'] for ind in population)} kg")

    # Store total biomass for plotting later
    total_biomass = sum(ind['biomass'] for ind in population)
    num_individuals_over_time.append(total_biomass)

    # Plotting code here (as before)
    plt.figure(figsize=(15, 7))

    # Plot population biomass density
    plt.subplot(1, 3, 1)
    plt.imshow(population_density, aspect='auto', cmap='inferno', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Biomass Density (kg)')
    plt.title('Biomass Density')

    # Plot adjusted resources
    plt.subplot(1, 3, 2)
    plt.imshow(adjusted_resources, aspect='auto', cmap='viridis', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Resources')
    plt.title('Adjusted Resources')

    # Plot land
    # Plot land
    plt.subplot(1, 3, 3)
    plt.imshow(land, aspect='auto', cmap='viridis', extent=[0, grid_size_lon, 0, grid_size_lat])
    plt.colorbar(label='Land')
    plt.title('Land')

    plt.show()
    
        
# Perform simulation
#simulation(population)

# # Plot oxygen gradient map as a blue shades heatmap over the land map
# plt.figure(figsize=(10, 5))
# plt.imshow(land, aspect='auto', cmap='Greys', extent=[0, grid_size_lon, 0, grid_size_lat])
# plt.imshow(oxygen_grid, aspect='auto', cmap='Blues', alpha=0.6, extent=[0, grid_size_lon, 0, grid_size_lat])
# plt.colorbar(label='Oxygen Level (%)')
# plt.title('Oxygen Gradient Map over Land')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

# # Plot temperature gradient map as a heatmap over the land map
# plt.figure(figsize=(10, 5))
# plt.imshow(land, aspect='auto', cmap='Greys', extent=[0, grid_size_lon, 0, grid_size_lat])
# plt.imshow(temperature_grid, aspect='auto', cmap='hot', alpha=0.6, extent=[0, grid_size_lon, 0, grid_size_lat])
# plt.colorbar(label='Temperature (°C)')
# plt.title('Temperature Gradient Map over Land')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()