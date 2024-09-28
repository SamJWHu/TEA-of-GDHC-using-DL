# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 04:26:29 2024

@author: SamJWHu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# Additional libraries
import seaborn as sns
from scipy import integrate


import pandas as pd
import numpy as np

# Generate depth values from 0 to 5000 m at 100 m intervals
depth = np.arange(0, 5001, 100)

# Simulate temperature gradient with some noise
temperature = 20 + 0.025 * depth + np.random.normal(0, 2, size=depth.shape)

# Simulate thermal conductivity, heat capacity, and rock density with slight variations
thermal_conductivity = 2.5 + 0.0001 * depth + np.random.normal(0, 0.1, size=depth.shape)
heat_capacity = 800 + 0.02 * depth + np.random.normal(0, 10, size=depth.shape)
rock_density = 2600 + 0.1 * depth + np.random.normal(0, 50, size=depth.shape)

# Create the DataFrame
geothermal_data_extended = pd.DataFrame({
    'Depth (m)': depth,
    'Temperature (°C)': temperature,
    'Thermal Conductivity (W/m·K)': thermal_conductivity,
    'Heat Capacity (J/kg·K)': heat_capacity,
    'Rock Density (kg/m³)': rock_density
})

# Save to CSV
geothermal_data_extended.to_csv('geothermal_data_extended.csv', index=False)



# Time array for two years
time = np.arange(0, 17520)

# Simulate operational parameters with variability
flow_rate = 50 + np.random.normal(0, 5, size=time.shape)  # kg/s
inlet_temp = 140 + np.random.normal(0, 5, size=time.shape)  # °C
outlet_temp = 70 + np.random.normal(0, 5, size=time.shape)  # °C

# Simulate outputs with added noise
electricity_output = 5000 + 100 * np.sin(2 * np.pi * time / 8760) + np.random.normal(0, 200, size=time.shape)  # kW
heat_output = 3500 + 100 * np.cos(2 * np.pi * time / 8760) + np.random.normal(0, 150, size=time.shape)  # kW

# Create the DataFrame
operational_data_extended = pd.DataFrame({
    'Time (hours)': time,
    'Flow Rate (kg/s)': flow_rate,
    'Inlet Temp (°C)': inlet_temp,
    'Outlet Temp (°C)': outlet_temp,
    'Electricity Output (kW)': electricity_output,
    'Heat Output (kW)': heat_output
})

# Save to CSV
operational_data_extended.to_csv('operational_data_extended.csv', index=False)




# Load the extended datasets
geothermal_data = pd.read_csv('geothermal_data_extended.csv')
operational_data = pd.read_csv('operational_data_extended.csv')


import pandas as pd

data = {
    'Item': [
        'Drilling', 'Plant Construction', 'Equipment', 'Permitting and Licensing',
        'Grid Connection', 'Contingency', 'Fixed Operational Costs', 'Variable Operational Costs',
        'Maintenance Costs', 'Administrative Costs', 'Electricity Sales', 'Heat Sales',
        'Renewable Energy Credits', 'Financial Parameters'
    ],
    'Capital Cost (USD)': [
        5000000, 10000000, 3000000, 500000, 1000000, 2000000, None, None, None, None, None, None, None, None
    ],
    'Operational Cost (USD/year)': [
        None, None, None, None, None, None, 300000, 0.02, 200000, 100000, None, None, None, None  # Variable cost as a fraction of revenue
    ],
    'Annual Revenue (USD/year)': [
        None, None, None, None, None, None, None, None, None, None, 2500000, 1500000, 500000, None
    ],
    'Project Lifetime (years)': [
        None, None, None, None, None, None, None, None, None, None, None, None, None, 30
    ],
    'Discount Rate (%)': [
        None, None, None, None, None, None, None, None, None, None, None, None, None, 7
    ]
}

economic_data_extended = pd.DataFrame(data)
economic_data_extended.to_csv('economic_data_extended.csv', index=False)



import pandas as pd

data = {
    'Material/Process': [
        'Steel Production', 'Concrete Production', 'Drilling Operations', 'Equipment Manufacturing',
        'Transportation', 'Chemical Usage', 'Waste Disposal', 'Operational Emissions'
    ],
    'Emission Factor (kg CO2-eq/unit)': [1.85, 0.13, 50, 2.5, 0.12, 5, 0.1, 0.05],
    'Energy Consumption (MJ/unit)': [20, 1.2, 600, 30, 1.5, 10, 0.5, 0.2],
    'Units': ['kg', 'kg', 'm', 'kg', 'ton·km', 'kg', 'kg', 'kWh']
}

environmental_data_extended = pd.DataFrame(data)
environmental_data_extended.to_csv('environmental_data_extended.csv', index=False)



import pandas as pd

# Load the extended datasets
economic_data = pd.read_csv('economic_data_extended.csv')
environmental_data = pd.read_csv('environmental_data_extended.csv')



from sklearn.model_selection import train_test_split


# Include 'Time (hours)' in features for splitting
features = operational_data[['Time (hours)', 'Flow Rate (kg/s)', 'Inlet Temp (°C)', 'Outlet Temp (°C)']]
targets = operational_data[['Electricity Output (kW)', 'Heat Output (kW)']]

# Separate 'Time (hours)' from features
time = features['Time (hours)']
X = features[['Flow Rate (kg/s)', 'Inlet Temp (°C)', 'Outlet Temp (°C)']]



from sklearn.model_selection import train_test_split

# First split off the test set
X_temp, X_test, y_temp, y_test, time_temp, time_test = train_test_split(
    X, targets, time, test_size=0.15, random_state=42)

# Then split the remaining data into training and validation sets
X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
    X_temp, y_temp, time_temp, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 ≈ 0.15



print(f'Training set size: {X_train.shape[0]} samples')
print(f'Validation set size: {X_val.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')



from sklearn.preprocessing import MinMaxScaler

# Initialize scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit the scalers on training data
X_train_scaled = feature_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train)

# Transform validation and test data
X_val_scaled = feature_scaler.transform(X_val)
y_val_scaled = target_scaler.transform(y_val)
X_test_scaled = feature_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test)

# Convert scaled data to float32
X_train_scaled = X_train_scaled.astype('float32')
y_train_scaled = y_train_scaled.astype('float32')
X_val_scaled = X_val_scaled.astype('float32')
y_val_scaled = y_val_scaled.astype('float32')
X_test_scaled = X_test_scaled.astype('float32')
y_test_scaled = y_test_scaled.astype('float32')


# Ensure inputs are float32
X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_scaled = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
X_val_scaled = tf.convert_to_tensor(X_val_scaled, dtype=tf.float32)
y_val_scaled = tf.convert_to_tensor(y_val_scaled, dtype=tf.float32)



import tensorflow as tf

def create_pinn(input_dim, output_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(50, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(50, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(output_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

pinn_model = create_pinn(input_dim=X_train_scaled.shape[1], output_dim=y_train_scaled.shape[1])



# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Initialize lists to store loss values
train_loss_history = []
val_loss_history = []

# Early stopping parameters
best_val_loss = np.inf
patience = 10
patience_counter = 0

# Training loop
epochs = 1000
batch_size = 64

# Convert data to tensors
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_scaled)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_scaled)).batch(batch_size)

for epoch in range(epochs):
    # Training step
    for X_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            predictions = pinn_model(X_batch, training=True)
            loss = tf.reduce_mean(tf.square(y_batch - predictions))
        gradients = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))

    # Calculate training loss
    train_predictions = pinn_model(X_train_scaled)
    train_loss = tf.reduce_mean(tf.square(y_train_scaled - train_predictions)).numpy()
    train_loss_history.append(train_loss)

    # Calculate validation loss
    val_predictions = pinn_model(X_val_scaled)
    val_loss = tf.reduce_mean(tf.square(y_val_scaled - val_predictions)).numpy()
    val_loss_history.append(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model weights with correct filename
        pinn_model.save_weights('best_pinn_model.weights.h5')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')




# Load the best model weights
pinn_model.load_weights('best_pinn_model.weights.h5')

pinn_model.save('best_pinn_model.keras')


import tensorflow as tf
print(tf.__version__)



# Inverse transform predictions and true values
test_predictions_scaled = pinn_model.predict(X_test_scaled)
test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
test_predictions = pd.DataFrame(test_predictions, columns=['Electricity Output (kW)', 'Heat Output (kW)'])
test_predictions = test_predictions.reset_index(drop=True)

# Ensure that y_test_inv is a DataFrame and reset index
y_test_inv = y_test.reset_index(drop=True)

# Retrieve the time indices for the test set and reset index
test_time = time_test.reset_index(drop=True)



plt.figure(figsize=(10,6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8,6))
plt.scatter(
    y_test_inv['Electricity Output (kW)'],
    test_predictions['Electricity Output (kW)'],
    alpha=0.5
)
plt.xlabel('True Electricity Output (kW)')
plt.ylabel('Predicted Electricity Output (kW)')
plt.title('Test Set: Predicted vs. True Electricity Output')
plt.plot(
    [y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
    [y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
    'r--'
)
plt.grid(True)
plt.show()





plt.figure(figsize=(8,6))
plt.scatter(
    y_test_inv['Heat Output (kW)'],
    test_predictions['Heat Output (kW)'],
    alpha=0.5
)
plt.xlabel('True Heat Output (kW)')
plt.ylabel('Predicted Heat Output (kW)')
plt.title('Test Set: Predicted vs. True Heat Output')
plt.plot(
    [y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
    [y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
    'r--'
)
plt.grid(True)
plt.show()



# Calculate residuals
residuals_electricity = y_test_inv['Electricity Output (kW)'] - test_predictions['Electricity Output (kW)']
residuals_heat = y_test_inv['Heat Output (kW)'] - test_predictions['Heat Output (kW)']

# Now you can proceed to analyze the residuals
# For example, plot residuals vs. predicted values
plt.figure(figsize=(8,6))
plt.scatter(test_predictions['Electricity Output (kW)'], residuals_electricity, alpha=0.5)
plt.xlabel('Predicted Electricity Output (kW)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals vs. Predicted Electricity Output')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()



# Heat Output Residuals
plt.figure(figsize=(8,6))
plt.scatter(
    test_predictions['Heat Output (kW)'],
    residuals_heat,
    alpha=0.5
)
plt.xlabel('Predicted Heat Output (kW)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Test Set Residuals vs. Predicted Heat Output')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()



# Electricity Output Residuals Histogram
plt.figure(figsize=(8,6))
plt.hist(residuals_electricity, bins=50, alpha=0.7, color='blue')
plt.xlabel('Residuals (kW)')
plt.ylabel('Frequency')
plt.title('Test Set Residuals Distribution - Electricity Output')
plt.grid(True)
plt.show()



# Heat Output Residuals Histogram
plt.figure(figsize=(8,6))
plt.hist(residuals_heat, bins=50, alpha=0.7, color='green')
plt.xlabel('Residuals (kW)')
plt.ylabel('Frequency')
plt.title('Test Set Residuals Distribution - Heat Output')
plt.grid(True)
plt.show()




# Ensure that the indices align
test_time = test_time.reset_index(drop=True)
y_test_inv = y_test_inv.reset_index(drop=True)
test_predictions = pd.DataFrame(test_predictions, columns=['Electricity Output (kW)', 'Heat Output (kW)'])
test_predictions = test_predictions.reset_index(drop=True)



# Plotting for a subset (e.g., first 168 samples in the test set)
subset_indices = np.argsort(test_time.values)[:168]

plt.figure(figsize=(12,6))
plt.plot(test_time.iloc[subset_indices], y_test_inv.iloc[subset_indices, 0], label='True Electricity Output')
plt.plot(test_time.iloc[subset_indices], test_predictions.iloc[subset_indices, 0], label='Predicted Electricity Output', linestyle='--')
plt.xlabel('Time (hours)')
plt.ylabel('Electricity Output (kW)')
plt.title('Test Set Electricity Output Over Time')
plt.legend()
plt.grid(True)
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error

# Electricity Output Metrics
mae_electricity = mean_absolute_error(y_test_inv.iloc[:, 0], test_predictions.iloc[:, 0])
rmse_electricity = np.sqrt(mean_squared_error(y_test_inv.iloc[:, 0], test_predictions.iloc[:, 0]))

print(f'Test Set - Electricity Output MAE: {mae_electricity:.2f} kW')
print(f'Test Set - Electricity Output RMSE: {rmse_electricity:.2f} kW')

# Heat Output Metrics
mae_heat = mean_absolute_error(y_test_inv.iloc[:, 1], test_predictions.iloc[:, 1])
rmse_heat = np.sqrt(mean_squared_error(y_test_inv.iloc[:, 1], test_predictions.iloc[:, 1]))

print(f'Test Set - Heat Output MAE: {mae_heat:.2f} kW')
print(f'Test Set - Heat Output RMSE: {rmse_heat:.2f} kW')



# Time array for one year
time_val = np.arange(0, 8760)

# Simulate operational parameters with different variability
flow_rate_val = 55 + np.random.normal(0, 5, size=time_val.shape)  # kg/s
inlet_temp_val = 145 + np.random.normal(0, 5, size=time_val.shape)  # °C
outlet_temp_val = 75 + np.random.normal(0, 5, size=time_val.shape)  # °C

# Simulate outputs with added noise
electricity_output_val = 5200 + 100 * np.sin(2 * np.pi * time_val / 8760) + np.random.normal(0, 200, size=time_val.shape)  # kW
heat_output_val = 3600 + 100 * np.cos(2 * np.pi * time_val / 8760) + np.random.normal(0, 150, size=time_val.shape)  # kW

# Create the DataFrame
operational_data_validation = pd.DataFrame({
    'Time (hours)': time_val,
    'Flow Rate (kg/s)': flow_rate_val,
    'Inlet Temp (°C)': inlet_temp_val,
    'Outlet Temp (°C)': outlet_temp_val,
    'Electricity Output (kW)': electricity_output_val,
    'Heat Output (kW)': heat_output_val
})

# Save to CSV
operational_data_validation.to_csv('operational_data_validation.csv', index=False)



# Load the new validation data
operational_data_val = pd.read_csv('operational_data_validation.csv')

# Features and targets
X_new_val = operational_data_val[['Flow Rate (kg/s)', 'Inlet Temp (°C)', 'Outlet Temp (°C)']]
y_new_val = operational_data_val[['Electricity Output (kW)', 'Heat Output (kW)']]

# Scale using the same scaler fitted on training data
X_new_val_scaled = feature_scaler.transform(X_new_val)
y_new_val_scaled = target_scaler.transform(y_new_val)

# Make predictions
new_val_predictions_scaled = pinn_model.predict(X_new_val_scaled)
new_val_predictions = target_scaler.inverse_transform(new_val_predictions_scaled)
y_new_val_inv = y_new_val.values  # Original values

# Evaluate performance
mae_electricity_new_val = mean_absolute_error(y_new_val_inv[:,0], new_val_predictions[:,0])
rmse_electricity_new_val = np.sqrt(mean_squared_error(y_new_val_inv[:,0], new_val_predictions[:,0]))

mae_heat_new_val = mean_absolute_error(y_new_val_inv[:,1], new_val_predictions[:,1])
rmse_heat_new_val = np.sqrt(mean_squared_error(y_new_val_inv[:,1], new_val_predictions[:,1]))

print(f'New Validation Set - Electricity Output MAE: {mae_electricity_new_val:.2f} kW')
print(f'New Validation Set - Electricity Output RMSE: {rmse_electricity_new_val:.2f} kW')
print(f'New Validation Set - Heat Output MAE: {mae_heat_new_val:.2f} kW')
print(f'New Validation Set - Heat Output RMSE: {rmse_heat_new_val:.2f} kW')




# Electricity Output
plt.figure(figsize=(8,6))
plt.scatter(y_new_val_inv[:,0], new_val_predictions[:,0], alpha=0.5)
plt.xlabel('True Electricity Output (kW)')
plt.ylabel('Predicted Electricity Output (kW)')
plt.title('New Validation Set: Predicted vs. True Electricity Output')
plt.plot([y_new_val_inv[:,0].min(), y_new_val_inv[:,0].max()],
         [y_new_val_inv[:,0].min(), y_new_val_inv[:,0].max()],
         'r--')
plt.grid(True)
plt.show()

# Heat Output
plt.figure(figsize=(8,6))
plt.scatter(y_new_val_inv[:,1], new_val_predictions[:,1], alpha=0.5)
plt.xlabel('True Heat Output (kW)')
plt.ylabel('Predicted Heat Output (kW)')
plt.title('New Validation Set: Predicted vs. True Heat Output')
plt.plot([y_new_val_inv[:,1].min(), y_new_val_inv[:,1].max()],
         [y_new_val_inv[:,1].min(), y_new_val_inv[:,1].max()],
         'r--')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()


# Load the best model weights
pinn_model.load_weights('best_pinn_model.weights.h5')

# Evaluate on test set
test_predictions_scaled = pinn_model.predict(X_test_scaled)
test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
test_predictions = pd.DataFrame(test_predictions, columns=['Electricity Output (kW)', 'Heat Output (kW)'])
test_predictions.reset_index(drop=True, inplace=True)
y_test_inv = y_test.reset_index(drop=True)



# Electricity Output
plt.figure(figsize=(8,6))
plt.scatter(y_test_inv['Electricity Output (kW)'], test_predictions['Electricity Output (kW)'], alpha=0.5)
plt.xlabel('True Electricity Output (kW)')
plt.ylabel('Predicted Electricity Output (kW)')
plt.title('Predicted vs. True Electricity Output')
plt.plot([y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
         [y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
         'r--')
plt.grid(True)
plt.show()

# Heat Output
plt.figure(figsize=(8,6))
plt.scatter(y_test_inv['Heat Output (kW)'], test_predictions['Heat Output (kW)'], alpha=0.5)
plt.xlabel('True Heat Output (kW)')
plt.ylabel('Predicted Heat Output (kW)')
plt.title('Predicted vs. True Heat Output')
plt.plot([y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
         [y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
         'r--')
plt.grid(True)
plt.show()



# Residuals
residuals_electricity = y_test_inv['Electricity Output (kW)'] - test_predictions['Electricity Output (kW)']
residuals_heat = y_test_inv['Heat Output (kW)'] - test_predictions['Heat Output (kW)']

# Electricity Output Residuals
plt.figure(figsize=(8,6))
plt.scatter(test_predictions['Electricity Output (kW)'], residuals_electricity, alpha=0.5)
plt.xlabel('Predicted Electricity Output (kW)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals vs. Predicted Electricity Output')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()

# Heat Output Residuals
plt.figure(figsize=(8,6))
plt.scatter(test_predictions['Heat Output (kW)'], residuals_heat, alpha=0.5)
plt.xlabel('Predicted Heat Output (kW)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residuals vs. Predicted Heat Output')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()




# Capital costs
capital_costs = economic_data['Capital Cost (USD)'].dropna().sum()

# Operational costs
fixed_operational_cost = economic_data.loc[economic_data['Item'] == 'Fixed Operational Costs', 'Operational Cost (USD/year)'].values[0]
variable_operational_cost_rate = economic_data.loc[economic_data['Item'] == 'Variable Operational Costs', 'Operational Cost (USD/year)'].values[0]

# Maintenance and administrative costs
maintenance_cost = economic_data.loc[economic_data['Item'] == 'Maintenance Costs', 'Operational Cost (USD/year)'].values[0]
administrative_cost = economic_data.loc[economic_data['Item'] == 'Administrative Costs', 'Operational Cost (USD/year)'].values[0]

# Total annual fixed costs
total_fixed_costs = fixed_operational_cost + maintenance_cost + administrative_cost

# Revenue
electricity_sales = economic_data.loc[economic_data['Item'] == 'Electricity Sales', 'Annual Revenue (USD/year)'].values[0]
heat_sales = economic_data.loc[economic_data['Item'] == 'Heat Sales', 'Annual Revenue (USD/year)'].values[0]
renewable_credits = economic_data.loc[economic_data['Item'] == 'Renewable Energy Credits', 'Annual Revenue (USD/year)'].values[0]

total_annual_revenue = electricity_sales + heat_sales + renewable_credits

# Financial parameters
project_lifetime = int(economic_data.loc[economic_data['Item'] == 'Financial Parameters', 'Project Lifetime (years)'].values[0])
discount_rate = float(economic_data.loc[economic_data['Item'] == 'Financial Parameters', 'Discount Rate (%)'].values[0]) / 100



# Variable operational costs depend on revenue (assuming a rate)
variable_operational_cost = variable_operational_cost_rate * total_annual_revenue

# Annual operational cost
annual_operational_cost = total_fixed_costs + variable_operational_cost

# Total lifetime costs
total_lifetime_costs = capital_costs + (annual_operational_cost * project_lifetime)



# Total electricity and heat outputs over project lifetime
total_lifetime_electricity = operational_data['Electricity Output (kW)'].sum() * project_lifetime / (1000 * 8760)  # Convert kW to GWh
total_lifetime_heat = operational_data['Heat Output (kW)'].sum() * project_lifetime / (1000 * 8760)  # Convert kW to GWh



lcoe = total_lifetime_costs / (total_lifetime_electricity * 1e6)  # USD per kWh
lcoh = total_lifetime_costs / (total_lifetime_heat * 1e6)  # USD per kWh

print(f'Levelized Cost of Electricity (LCOE): ${lcoe:.4f} per kWh')
print(f'Levelized Cost of Heat (LCOH): ${lcoh:.4f} per kWh')



# Quantities of materials used (assumed values)
materials_used = {
    'Steel Production': 80000,  # kg
    'Concrete Production': 150000,  # kg
    'Drilling Operations': 5000,  # m
    'Equipment Manufacturing': 30000,  # kg
    'Transportation': 1500,  # ton·km
    'Chemical Usage': 10000,  # kg
    'Waste Disposal': 5000,  # kg
    'Operational Emissions': operational_data['Electricity Output (kW)'].sum() * project_lifetime / 1000  # MWh
}

total_emissions = 0
emissions_breakdown = {}

for material, quantity in materials_used.items():
    emission_factor = environmental_data.loc[environmental_data['Material/Process'] == material, 'Emission Factor (kg CO2-eq/unit)'].values[0]
    emissions = quantity * emission_factor
    total_emissions += emissions
    emissions_breakdown[material] = emissions

print(f'Total GHG Emissions: {total_emissions:,.2f} kg CO2-eq')



# Total electricity generated over project lifetime in kWh
total_electricity_generated = operational_data['Electricity Output (kW)'].sum() * project_lifetime

emissions_per_kwh = total_emissions / total_electricity_generated
print(f'GHG Emissions per kWh: {emissions_per_kwh:.4f} kg CO2-eq/kWh')




labels = ['LCOE', 'LCOH']
costs = [lcoe, lcoh]

plt.figure(figsize=(6,4))
plt.bar(labels, costs, color=['blue', 'green'])
plt.ylabel('Cost (USD per kWh)')
plt.title('Levelized Costs')
plt.show()



materials = list(emissions_breakdown.keys())
emissions_values = list(emissions_breakdown.values())

plt.figure(figsize=(10,6))
plt.barh(materials, emissions_values)
plt.xlabel('GHG Emissions (kg CO2-eq)')
plt.title('GHG Emissions by Material/Process')
plt.tight_layout()
plt.show()




import numpy_financial as npf

# Initial investment
initial_investment = capital_costs

# Annual net cash flow
annual_cash_flow = total_annual_revenue - annual_operational_cost

# Cash flows list
cash_flows = [-initial_investment] + [annual_cash_flow] * project_lifetime

# Cumulative cash flow
cumulative_cash_flows = np.cumsum(cash_flows)

# Plot
years = np.arange(project_lifetime + 1)

plt.figure(figsize=(10,6))
plt.plot(years, cumulative_cash_flows, marker='o')
plt.xlabel('Year')
plt.ylabel('Cumulative Cash Flow (USD)')
plt.title('Cumulative Cash Flow Over Project Lifetime')
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()



# Net Present Value
npv = npf.npv(discount_rate, cash_flows)
print(f'Net Present Value (NPV): ${npv:,.2f}')

# Internal Rate of Return
irr = npf.irr(cash_flows)
print(f'Internal Rate of Return (IRR): {irr * 100:.2f}%')

# Payback Period
cumulative_cash_flows = np.cumsum(cash_flows)
payback_period = next((i for i, x in enumerate(cumulative_cash_flows) if x > 0), None)
if payback_period is not None:
    print(f'Payback Period: {payback_period} years')
else:
    print('The project does not pay back within the project lifetime.')



# Vary discount rate from 5% to 15%
discount_rates = np.linspace(0.05, 0.15, 11)
npv_values = []

for dr in discount_rates:
    npv_value = npf.npv(dr, cash_flows)
    npv_values.append(npv_value)

plt.figure(figsize=(8,6))
plt.plot(discount_rates * 100, npv_values)
plt.xlabel('Discount Rate (%)')
plt.ylabel('NPV (USD)')
plt.title('NPV vs. Discount Rate')
plt.grid(True)
plt.show()







