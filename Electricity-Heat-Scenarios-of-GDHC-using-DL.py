# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 04:55:38 2024

@author: SamJWHu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# Additional libraries
import seaborn as sns
from scipy import integrate


import numpy as np
import pandas as pd

# Time array for two years
time = np.arange(0, 17520)  # 2 years of hourly data

# Define multiple operational scenarios
scenarios = [
    {
        'Flow Rate (kg/s)': lambda t: 50 + 5 * np.sin(2 * np.pi * t / 8760),  # Seasonal variation
        'Inlet Temp (°C)': lambda t: 140 + 10 * np.cos(2 * np.pi * t / 8760),
        'Outlet Temp (°C)': lambda t: 70 + 5 * np.sin(4 * np.pi * t / 8760),
        'Electricity Output (kW)': lambda t: 5000 + 200 * np.sin(2 * np.pi * t / 8760),
        'Heat Output (kW)': lambda t: 3500 + 150 * np.cos(2 * np.pi * t / 8760)
    },
    {
        'Flow Rate (kg/s)': lambda t: 55 + 5 * np.sin(2 * np.pi * t / 8760 + np.pi / 4),
        'Inlet Temp (°C)': lambda t: 145 + 8 * np.cos(2 * np.pi * t / 8760 + np.pi / 4),
        'Outlet Temp (°C)': lambda t: 75 + 5 * np.sin(4 * np.pi * t / 8760 + np.pi / 4),
        'Electricity Output (kW)': lambda t: 5200 + 180 * np.sin(2 * np.pi * t / 8760 + np.pi / 4),
        'Heat Output (kW)': lambda t: 3600 + 130 * np.cos(2 * np.pi * t / 8760 + np.pi / 4)
    },
    # Add more scenarios as needed
]



# Initialize a list to hold data from all scenarios
operational_data_list = []

for idx, scenario in enumerate(scenarios):
    scenario_data = pd.DataFrame({'Time (hours)': time})
    for key, func in scenario.items():
        if key != 'Time (hours)':
            scenario_data[key] = func(time) + np.random.normal(0, 5, size=time.shape)
    scenario_data['Scenario'] = idx + 1
    operational_data_list.append(scenario_data)

# Combine all scenarios into a single DataFrame
operational_data_extended = pd.concat(operational_data_list, ignore_index=True)


# Save to CSV
operational_data_extended.to_csv('operational_data_extended.csv', index=False)


# Load the extended operational data
operational_data = pd.read_csv('operational_data_extended.csv')



from sklearn.model_selection import train_test_split

# Features and targets
features = operational_data[['Flow Rate (kg/s)', 'Inlet Temp (°C)', 'Outlet Temp (°C)', 'Scenario']]
targets = operational_data[['Electricity Output (kW)', 'Heat Output (kW)']]
time = operational_data['Time (hours)']

# First, split off the test set (15%)
X_temp, X_test, y_temp, y_test, time_temp, time_test = train_test_split(
    features, targets, time,
    test_size=0.15,
    random_state=42,
    stratify=operational_data['Scenario']
)

# Then split the remaining data into training and validation sets (85% * 17.65% ≈ 15% validation)
X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
    X_temp, y_temp, time_temp,
    test_size=0.1765,
    random_state=42,
    stratify=X_temp['Scenario']
)



# After splitting the data

# Retrieve 'Scenario' information from X_test before dropping the column
scenario_test = X_test['Scenario'].reset_index(drop=True)

# Retrieve 'Scenario' information from X_train and X_val if needed
scenario_train = X_train['Scenario'].reset_index(drop=True)
scenario_val = X_val['Scenario'].reset_index(drop=True)

# Remove 'Scenario' column from features
X_train = X_train.drop(columns=['Scenario'])
X_val = X_val.drop(columns=['Scenario'])
X_test = X_test.drop(columns=['Scenario'])




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

# Convert to float32
X_train_scaled = X_train_scaled.astype('float32')
y_train_scaled = y_train_scaled.astype('float32')
X_val_scaled = X_val_scaled.astype('float32')
y_val_scaled = y_val_scaled.astype('float32')
X_test_scaled = X_test_scaled.astype('float32')
y_test_scaled = y_test_scaled.astype('float32')




import tensorflow as tf

def create_pinn(input_dim, output_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(output_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

pinn_model = create_pinn(input_dim=X_train_scaled.shape[1], output_dim=y_train_scaled.shape[1])




optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
pinn_model.compile(optimizer=optimizer, loss='mse')



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_pinn_model.keras', monitor='val_loss', save_best_only=True)
]



history = pinn_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=1000,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# Replace the existing model with the loaded model
pinn_model = tf.keras.models.load_model('best_pinn_model.keras')



test_predictions_scaled = pinn_model.predict(X_test_scaled)
test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
test_predictions = pd.DataFrame(test_predictions, columns=['Electricity Output (kW)', 'Heat Output (kW)'])
test_predictions.reset_index(drop=True, inplace=True)
y_test_inv = y_test.reset_index(drop=True)
time_test = time_test.reset_index(drop=True)



from sklearn.metrics import mean_absolute_error, mean_squared_error

# Electricity Output Metrics
mae_electricity = mean_absolute_error(y_test_inv['Electricity Output (kW)'], test_predictions['Electricity Output (kW)'])
rmse_electricity = np.sqrt(mean_squared_error(y_test_inv['Electricity Output (kW)'], test_predictions['Electricity Output (kW)']))

print(f'Test Set - Electricity Output MAE: {mae_electricity:.2f} kW')
print(f'Test Set - Electricity Output RMSE: {rmse_electricity:.2f} kW')

# Heat Output Metrics
mae_heat = mean_absolute_error(y_test_inv['Heat Output (kW)'], test_predictions['Heat Output (kW)'])
rmse_heat = np.sqrt(mean_squared_error(y_test_inv['Heat Output (kW)'], test_predictions['Heat Output (kW)']))

print(f'Test Set - Heat Output MAE: {mae_heat:.2f} kW')
print(f'Test Set - Heat Output RMSE: {rmse_heat:.2f} kW')



import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8,6))
plt.scatter(y_test_inv['Electricity Output (kW)'], test_predictions['Electricity Output (kW)'], alpha=0.5)
plt.xlabel('True Electricity Output (kW)')
plt.ylabel('Predicted Electricity Output (kW)')
plt.title('Predicted vs. True Electricity Output')
plt.plot(
    [y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
    [y_test_inv['Electricity Output (kW)'].min(), y_test_inv['Electricity Output (kW)'].max()],
    'r--'
)
plt.grid(True)
plt.show()



plt.figure(figsize=(8,6))
plt.scatter(y_test_inv['Heat Output (kW)'], test_predictions['Heat Output (kW)'], alpha=0.5)
plt.xlabel('True Heat Output (kW)')
plt.ylabel('Predicted Heat Output (kW)')
plt.title('Predicted vs. True Heat Output')
plt.plot(
    [y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
    [y_test_inv['Heat Output (kW)'].min(), y_test_inv['Heat Output (kW)'].max()],
    'r--'
)
plt.grid(True)
plt.show()



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



# Electricity Output Residuals Histogram
plt.figure(figsize=(8,6))
plt.hist(residuals_electricity, bins=50, alpha=0.7, color='blue')
plt.xlabel('Residuals (kW)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution - Electricity Output')
plt.grid(True)
plt.show()

# Heat Output Residuals Histogram
plt.figure(figsize=(8,6))
plt.hist(residuals_heat, bins=50, alpha=0.7, color='green')
plt.xlabel('Residuals (kW)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution - Heat Output')
plt.grid(True)
plt.show()




# Example usage
print(scenario_test.head())



import seaborn as sns

# Combine residuals and scenario into a DataFrame
residuals_df = pd.DataFrame({
    'Electricity Residuals': residuals_electricity,
    'Heat Residuals': residuals_heat,
    'Scenario': scenario_test
})

# Electricity Residuals Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x='Scenario', y='Electricity Residuals', data=residuals_df)
plt.title('Electricity Output Residuals by Scenario')
plt.grid(True)
plt.show()

# Heat Residuals Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x='Scenario', y='Heat Residuals', data=residuals_df)
plt.title('Heat Output Residuals by Scenario')
plt.grid(True)
plt.show()



# Select a scenario to visualize
selected_scenario = 1

# Filter data for the selected scenario
indices = scenario_test[scenario_test == selected_scenario].index
time_scenario = time_test.iloc[indices]
y_true_scenario = y_test_inv.iloc[indices]
y_pred_scenario = test_predictions.iloc[indices]

# Sort by time
sorted_indices = time_scenario.argsort()
time_scenario = time_scenario.iloc[sorted_indices]
y_true_scenario = y_true_scenario.iloc[sorted_indices]
y_pred_scenario = y_pred_scenario.iloc[sorted_indices]

# Plot Electricity Output over time
plt.figure(figsize=(12,6))
plt.plot(time_scenario, y_true_scenario['Electricity Output (kW)'], label='True Electricity Output')
plt.plot(time_scenario, y_pred_scenario['Electricity Output (kW)'], label='Predicted Electricity Output', linestyle='--')
plt.xlabel('Time (hours)')
plt.ylabel('Electricity Output (kW)')
plt.title(f'Scenario {selected_scenario} Electricity Output Over Time')
plt.legend()
plt.grid(True)
plt.show()



# Select a scenario to visualize
selected_scenario = 2

# Filter data for the selected scenario
indices = scenario_test[scenario_test == selected_scenario].index
time_scenario = time_test.iloc[indices]
y_true_scenario = y_test_inv.iloc[indices]
y_pred_scenario = test_predictions.iloc[indices]

# Sort by time
sorted_indices = time_scenario.argsort()
time_scenario = time_scenario.iloc[sorted_indices]
y_true_scenario = y_true_scenario.iloc[sorted_indices]
y_pred_scenario = y_pred_scenario.iloc[sorted_indices]

# Plot Electricity Output over time
plt.figure(figsize=(12,6))
plt.plot(time_scenario, y_true_scenario['Electricity Output (kW)'], label='True Electricity Output')
plt.plot(time_scenario, y_pred_scenario['Electricity Output (kW)'], label='Predicted Electricity Output', linestyle='--')
plt.xlabel('Time (hours)')
plt.ylabel('Electricity Output (kW)')
plt.title(f'Scenario {selected_scenario} Electricity Output Over Time')
plt.legend()
plt.grid(True)
plt.show()





