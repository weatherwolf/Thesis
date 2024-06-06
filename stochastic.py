# %%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, truncnorm
import datetime as date
from scipy.stats import norm
from data import *

def inTimeIntervals(time, intervals):
    index = 0
    for interval in intervals:
        if time >= interval[0] and time <= interval[1]:
            index = intervals.index(interval)

    return index

# %%
# Load dataset
data = pd.read_csv('Data_Madrid_2011-2016.csv')

# Convert the 'time' column to datetime
data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')

# Set the 'time' column as the index
data.set_index('time', inplace=True)

# Rename columns for clarity
data.columns = ['pv_generation', 'H_sun', 'temperature', 'WS10m', 'Int']

# Resample to 10-minute intervals using interpolation
data_10min = data.resample('10T').interpolate(method='linear')

# Function to add small noise
def add_noise(series, noise_level=0.01):
    noise = np.random.normal(0, noise_level, series.shape)
    return series + noise

def fix_series(series, threshold=0.1):
    series[np.abs(series) <= threshold] = 0
    return series

# Example usage
# data_10min is assumed to be your input series
data_10min = add_noise(data_10min)
data_10min = fix_series(data_10min)

print(data_10min.head())
print(np.shape(data_10min))

temperature = {}
pv_generation = {}


startdate = date.datetime(2011, 1, 1, 0, 0)

i = 0
while i <= np.shape(data_10min)[0] - 24*6:
    temperature_day = []
    pv_generation_day = []
    for j in range(24*6):
        temperature_day.append(data_10min['temperature'][i])
        pv_generation_day.append(data_10min['pv_generation'][i])
        i += 1
    name = f'{startdate.year}-{startdate.month}-{startdate.day}'
    startdate += date.timedelta(days=1)

    temperature[name] = temperature_day
    pv_generation[name] = pv_generation_day



# # %%
# random_date = "2012-4-20"
# test = pd.DataFrame(temperature[random_date])
# plt.plot(test)
# # plt.plot(pv_generation[random_date])

# # %%
# # Calculate monthly statistics
# monthly_stats = data_10min.groupby(data_10min.index.month).agg(['mean', 'std'])

# # Function to generate diurnal cycle
# def diurnal_cycle(hours):
#     return np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 2 

# # Function to generate temperature cycle
# def temperature_cycle(hours, randomness=0.0):
#     return (5 + randomness/10) * np.sin(np.pi * (hours - 8) / 12)

# # Function to generate synthetic data for a month
# def generate_synthetic_data(month, num_days=1, num_intervals_per_day=144):
#     temperature_stats = monthly_stats.loc[month, ('temperature', 'mean')], monthly_stats.loc[month, ('temperature', 'std')]
#     pv_generation_stats = monthly_stats.loc[month, ('pv_generation', 'mean')], monthly_stats.loc[month, ('pv_generation', 'std')]

#     hours = np.tile(np.arange(0, 24, 24 / num_intervals_per_day), num_days)
    
#     lowerbound = random.randint(-20,10)
#     upperbound = random.randint(11,40)

#     strenght = random.randint(-50,50)

#     temperature_pattern = temperature_cycle(hours, randomness=strenght)
    
#     # Calculate the number of intervals for the first hour
#     intervals_per_hour = num_intervals_per_day // 24
#     first_hour_intervals = intervals_per_hour

#     # Generate temperature data with reduced randomness for the first hour
#     reduced_std = temperature_stats[1] / 2  # Reduce the standard deviation by half for the first hour
#     synthetic_temperature = np.zeros(num_days * num_intervals_per_day)
    
#     for i in range(num_days):
#         # Generate temperature for the first hour with reduced randomness
#         synthetic_temperature[i*num_intervals_per_day : i*num_intervals_per_day + first_hour_intervals] = temperature_pattern[i*num_intervals_per_day : i*num_intervals_per_day + first_hour_intervals] + truncnorm.rvs(
#             (lowerbound - temperature_stats[0]) / reduced_std,
#             (upperbound - temperature_stats[0]) / reduced_std,
#             loc=temperature_stats[0], scale=reduced_std, size=first_hour_intervals
#         )
        
#         # Generate temperature for the rest of the day with normal randomness
#         synthetic_temperature[i*num_intervals_per_day + first_hour_intervals : (i+1)*num_intervals_per_day] = temperature_pattern[i*num_intervals_per_day + first_hour_intervals : (i+1)*num_intervals_per_day] + truncnorm.rvs(
#             (lowerbound - temperature_stats[0]) / temperature_stats[1],
#             (upperbound - temperature_stats[0]) / temperature_stats[1],
#             loc=temperature_stats[0], scale=temperature_stats[1], size=num_intervals_per_day - first_hour_intervals
#         )
    
#     diurnal_pattern = diurnal_cycle(hours)
#     synthetic_pv_generation = diurnal_pattern * truncnorm.rvs(
#         (0 - pv_generation_stats[0]) / pv_generation_stats[1],
#         (2000 - pv_generation_stats[0]) / pv_generation_stats[1],
#         loc=pv_generation_stats[0], scale=pv_generation_stats[1], size=num_days * num_intervals_per_day
#     )
    
#     time_index = pd.date_range(start=f'2012-{month:02d}-01', periods=num_days * num_intervals_per_day, freq='10T')

#     synthetic_data = pd.DataFrame({'temperature': synthetic_temperature, 'pv_generation': synthetic_pv_generation}, index=time_index)
    
#     return synthetic_data


# # %% 

# # Generate synthetic data for the first day of each month
# synthetic_data_by_month = {}
# for month in range(1, 13):
#     synthetic_data_by_month[month] = generate_synthetic_data(month)

# # Apply smoothing using a rolling window
# window_size = 10  # Adjust window size for desired smoothness
# for month in range(1, 13):
#     synthetic_data_by_month[month]['temperature'] = synthetic_data_by_month[month]['temperature'].rolling(window=window_size, min_periods=1).mean()
#     synthetic_data_by_month[month]['pv_generation'] = synthetic_data_by_month[month]['pv_generation'].rolling(window=window_size, min_periods=1).mean()

# # Example: Plot smoothed synthetic data for the first day of January
# synthetic_data_by_month[4].plot(subplots=True, figsize=(12, 8))
# plt.show()

# # Optionally, combine all synthetic data into a single DataFrame
# synthetic_data = pd.concat(synthetic_data_by_month.values())

# weather_data = {}
# for i in range(1000):
#     synthetic_data = generate_synthetic_data(4)
#     weather_data[i] = synthetic_data['temperature'].rolling(window=window_size, min_periods=1).mean()

# weather_data = pd.DataFrame(weather_data)

# plt.figure(figsize=(12, 6))

# for simulation in weather_data:
#     plt.plot(weather_data[simulation], color='gray', alpha=0.1)

# plt.show()
# # %%
# # Number of simulations
# num_simulations = 100

# # Store results for analysis
# simulation_results = []

# plt.figure(figsize=(12, 6))

# # Generate synthetic data for each simulation
# for sim in range(num_simulations):
#     monthly_data = {}
#     for month in range(1, 13):
#         monthly_data[month] = generate_synthetic_data(month)
#         if month == 4:
#             monthly_data[month]['temperature'] = monthly_data[month]['temperature'].rolling(window=window_size, min_periods=1).mean()
#             plt.plot(monthly_data[month]['temperature'], label=f'Month {month}', alpha=0.2, color='gray')
    
#     # Combine all monthly data into a single DataFrame
#     simulated_year_data = pd.concat(monthly_data.values())
    
#     # Store the results
#     simulation_results.append(simulated_year_data)

# plt.show()

# # %%
# plt.figure(figsize=(12, 6))
# for year in range(2011, 2017):
#     for day in range(1,30):
#         date = f'{year}-1-{day}'
#         plt.plot(temperature[date], label=date, alpha=0.1)

#         if temperature[date][20] > 11:
#             print(date)

# plt.show()
# # %%
# plt.figure(figsize=(12, 6))
# given_data = []
# for i in range(1440):
#     given_data.append(data_sm11["theta_t_ext"][inTimeIntervals(i, data_sm11["Time Interval"])])

# plt.plot(given_data)
# plt.show()

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

lag = 1

temperature_data = []
pv_generation_data = []
for year in range(2011, 2016):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-30'
    date_range = pd.date_range(start=start_date, end=end_date)
    for date in date_range:
        date_str = f'{date.year}-{date.month}-{date.day}'  # Manually format date string without leading zeros
        temperature_data.append(temperature[date_str])
        pv_generation_data.append(pv_generation[date_str])

def create_lagged_features(data, name, lags=lag):
    lagged_data = pd.DataFrame(data, columns=[name])
    for lag in range(1, lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data[name].shift(lag)
    lagged_data = lagged_data.dropna()
    return lagged_data

# Create lagged features
temperature_data_lagged = []
for sim_data in temperature_data:
    temperature_data_lagged.append(create_lagged_features(sim_data, 'temperature'))

pv_generation_data_lagged = []
for sim_data in pv_generation_data:
    pv_generation_data_lagged.append(create_lagged_features(sim_data, 'pv generation'))

# Concatenate all lagged features into a single DataFrame
temperature_data_lagged = pd.concat(temperature_data_lagged, ignore_index=True)

# Prepare the features and target
time = np.arange(len(temperature_data_lagged))
features_temperature = pd.DataFrame({
    'time': time,
    **{f'lag_{i}': temperature_data_lagged[f'lag_{i}'] for i in range(1, lag + 1)}
})

pv_generation_data_lagged = pd.concat(pv_generation_data_lagged, ignore_index=True)

# Prepare the features and target
time = np.arange(len(pv_generation_data_lagged))
features_pv_generation = pd.DataFrame({
    'time': time,
    **{f'lag_{i}': pv_generation_data_lagged[f'lag_{i}'] for i in range(1, lag + 1)}
})

    
# The target is the future temperature, which could be the temperature at the current time
target_temperature = temperature_data_lagged['temperature']
target_pv_generation = pv_generation_data_lagged['pv generation']

# %%

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_temperature, target_temperature, test_size=0.2, shuffle=False)


# Train a Random Forest Regressor model
model_temperature = RandomForestRegressor(n_estimators=100, random_state=42)
model_temperature.fit(X_train, y_train)

# %% 
# Predict on the test set
y_outlier_test = temperature['2016-6-10']
time_values = list(range(len(y_outlier_test)))
x_outlier_test = create_lagged_features(y_outlier_test, 'temperature', lag)
x_outlier_test.drop('temperature', axis=1, inplace=True)
x_outlier_test['time'] = time_values[lag:]

cols = x_outlier_test.columns.tolist()
cols = ['time'] + [col for col in cols if col != 'time']
x_outlier_test = x_outlier_test[cols]

# Predict on the test set
y_pred = model_temperature.predict(x_outlier_test)
y_test = y_outlier_test[lag:]

mse = mean_squared_error(y_test, y_pred)

# Plot the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(y_pred, label='Predicted Temperature', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperature using Random Forest')
plt.legend()
plt.show()

### PV_GENERATION ###
# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_pv_generation, target_pv_generation, test_size=0.2, shuffle=False)


# Train a Random Forest Regressor model
model_pv_generation = RandomForestRegressor(n_estimators=100, random_state=42)
model_pv_generation.fit(X_train, y_train)

# %% 
# Predict on the test set
y_pred = model_pv_generation.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
y_pred = pd.Series(y_pred, index=y_test.index)

# Plot the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(y_test[0:24*6], label='Actual Temperature')
plt.plot(y_pred[0:24*6], label='Predicted Temperature', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperature using Random Forest')
plt.legend()
plt.show()


# %%
joblib.dump(model_temperature, 'random_forest_model_temperature.pkl')
joblib.dump(model_pv_generation, 'random_forest_model_pv_generation.pkl')

# %%
