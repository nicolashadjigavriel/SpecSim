import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 1    # Exponent for oxygen in metabolic rate equation
beta = 0.06  # Temperature coefficient
gamma = 2    # Steepness of the logistic curve
theta = 1    # Threshold ratio where survival probability is 0.5
I_rate = 0.03  # Intake rate in kg/day
pO2 = 21
T = 20


# Load the data
data = pd.read_csv('pantheria_filtered_data.csv')

# Filter the DataFrame for diet values less than 3
herbivore_data = data[data['6-2_TrophicLevel'] < 2]

# Extract the relevant columns from the herbivore_data
R_d = herbivore_data['18-1_BasalMetRate_mLO2hr']
M = herbivore_data['5-1_AdultBodyMass_g']
T = herbivore_data['28-2_Temp_Mean_01degC'] / 10
Lon = herbivore_data['26-7_GR_MRLong_dd']
Lat = herbivore_data['26-4_GR_MRLat_dd']
# Generate NPP and adjust it to match the size of M
NPP = np.linspace(0, 3000, len(M))  # Adjust NPP to the size of M

# Function to calculate metabolic rate R_d
def metabolic_rate(M, T, pO2, alpha, beta):
    return 3.84 * (M**(0.7)) * (pO2**alpha) * np.exp(beta * T)

# Function to calculate daily intake I_d
def daily_intake(I_rate, M):
    return I_rate * M

# Function to calculate carrying capacity K
def carrying_capacity(NPP, I_d):
    return NPP / I_d

# Lotka-Volterra based survival probability
def survival_probability(K, R_d):
    return 1 / (1 + (R_d / K))

# Compute daily intake and carrying capacity
I_d = daily_intake(I_rate, M)
K = carrying_capacity(NPP, I_d)
S = survival_probability(K, R_d)
R_theoretical = metabolic_rate(M, T, pO2, alpha, beta)

# Plot Survival Probability vs BMR as a scatter plot
plt.figure()
plt.scatter(R_d, S, c='blue', marker='o', edgecolor='black')
plt.title("Survival Probability vs BMR (Lotka-Volterra)")
plt.xlabel("BMR")
plt.ylabel("Survival Probability")
plt.xlim([0, 1000])  # Set x-axis limits
plt.ylim([0, 1])  # Set y-axis limits (Survival Probability is typically between 0 and 1)
plt.grid(True)
plt.show()


# Plot Survival Probability vs Carrying Capacity K as a scatter plot
plt.figure()
plt.scatter(K, S, c='red', marker='o', edgecolor='black')
plt.title("Survival Probability vs Carrying Capacity K (Lotka-Volterra)")
plt.xlabel("K")
plt.ylabel("S")
plt.xlim([0, 1000])  # Set y-axis limits based on K
plt.ylim([0, 1])  # Set x-axis limits (Survival Probability is typically between 0 and 1)
plt.grid(True)
plt.show()

# Plot Survival Probability vs Mass in grams as a scatter plot
plt.figure()
plt.scatter(M, S, c='green', marker='o', edgecolor='black')
plt.title("Survival Probability vs Mass(g) (Lotka-Volterra)")
plt.xlabel("M(g)")
plt.ylabel("S")
plt.xlim([0, 1000])  # Set y-axis limits based on K
plt.ylim([0, 1])  # Set x-axis limits (Survival Probability is typically between 0 and 1)
plt.grid(True)
plt.show()

# 
plt.figure()
plt.scatter(R_d, R_theoretical, c='orange', marker='o', edgecolor='black')
plt.title("bmr from data vs bmr from equation")
plt.xlabel("bmr from data)")
plt.ylabel("bmr from equation")
plt.xlim([0, 300])  # Set y-axis limits based on K
plt.ylim([0, 10000])  # Set x-axis limits (Survival Probability is typically between 0 and 1)
plt.grid(True)
plt.show()


# Create scatter map and temperature
plt.scatter(Lon, Lat, c=T, cmap='coolwarm', marker='o')

# Adding labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('PanTheria Map of temperature')

# Adding colorbar to indicate temperature variation
cbar = plt.colorbar()
cbar.set_label('Temperature (°C)')

# Display plot
plt.show()


# Create scatter map and Mass
plt.scatter(Lon, Lat, c=M, cmap='coolwarm', marker='o')

# Adding labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('PanTheria Map of Mass')

# Adding colorbar to indicate temperature variation
cbar = plt.colorbar()
cbar.set_label('Mass (g)')

# Display plot
plt.show()

# Save the herbivore_data to a CSV file
herbivore_data.to_csv('herbivore_data.csv', index=False)
