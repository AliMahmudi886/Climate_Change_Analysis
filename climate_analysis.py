# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Step 1: Load the dataset
file_path = r'C:\climate_change_analysis\data\GlobalWeatherRepository.csv'
data = pd.read_csv(file_path)
print("Here's a preview of the data we're working with:")
print(data.head())  # Show the first few rows of the data

# Step 2: Handle missing values
print("\nChecking for any missing values in the data...")
print(data.isnull().sum())  # Display the number of missing values in each column

# Remove rows with missing values to clean up the dataset
df = data.dropna()

# Step 3: Extract Year and Month (if possible)
# Convert any date columns to proper datetime format
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
elif 'last_updated' in df.columns:
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['year'] = df['last_updated'].dt.year
    df['month'] = df['last_updated'].dt.month
else:
    print("No date columns found. We'll use the data as it is.")

# Step 4: Select useful features for our model
features = ['year', 'month', 'latitude', 'longitude', 'temperature_celsius', 
            'humidity', 'wind_kph', 'pressure_mb', 'precip_mm']

# Keep only the selected features
df = df[features].dropna()

# Step 5: Create new features to help the model learn better
# Categorize months into seasons: Winter, Spring, Summer, Fall
df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] 
                                 else 'Spring' if x in [3, 4, 5] 
                                 else 'Summer' if x in [6, 7, 8] 
                                 else 'Fall')

# Convert seasons into numbers (one-hot encoding)
df = pd.get_dummies(df, columns=['season'], drop_first=True)

# Create new features that combine existing ones
df['humidity_pressure_interaction'] = df['humidity'] * df['pressure_mb']
df['wind_precip_interaction'] = df['wind_kph'] * df['precip_mm']

# Step 6: Visualize correlations between features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('How different factors relate to each other (Correlation Heatmap)')
plt.show()

# Step 7: Prepare data for training (split into training and testing sets)
X = df.drop('temperature_celsius', axis=1)  # Features (everything except temperature)
y = df['temperature_celsius']  # Target variable (temperature)

# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Choose the models to train and their hyperparameters
# We'll try three models: Gradient Boosting, Random Forest, and XGBoost
models = {
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1]
        }
    }
}

# Step 9: Train each model and find the best settings
best_estimators = {}
for name, config in models.items():
    print(f"Finding the best settings for {name}...")
    grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search.best_estimator_
    print(f"Best settings for {name}: {grid_search.best_params_}")

# Step 10: Combine the best models into an ensemble model
print("\nCombining the best models into a single ensemble model...")
ensemble_model = VotingRegressor(estimators=[('gbr', best_estimators['Gradient Boosting']), 
                                             ('rf', best_estimators['Random Forest']), 
                                             ('xgb', best_estimators['XGBoost'])])
ensemble_model.fit(X_train, y_train)

# Step 11: Test the ensemble model and show results
print("\nTesting the ensemble model...")
y_pred = ensemble_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual Temperatures")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label="Perfect Prediction Line")
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("How Close the Predictions are to the Actual Temperatures")
plt.legend()
plt.show()

# Step 12: Predict future temperature trends
print("\nPredicting temperature trends for the next 10 years...")
future_years = np.arange(df['year'].max() + 1, df['year'].max() + 11)
future_data = pd.DataFrame({'year': future_years, 'month': [1] * len(future_years), 
                            'latitude': [df['latitude'].mean()] * len(future_years),
                            'longitude': [df['longitude'].mean()] * len(future_years),
                            'humidity': [df['humidity'].mean()] * len(future_years),
                            'wind_kph': [df['wind_kph'].mean()] * len(future_years),
                            'pressure_mb': [df['pressure_mb'].mean()] * len(future_years),
                            'precip_mm': [df['precip_mm'].mean()] * len(future_years),
                            'humidity_pressure_interaction': [df['humidity'].mean() * df['pressure_mb'].mean()] * len(future_years),
                            'wind_precip_interaction': [df['wind_kph'].mean() * df['precip_mm'].mean()] * len(future_years),
                            'season_Spring': [0] * len(future_years), 
                            'season_Summer': [0] * len(future_years), 
                            'season_Winter': [0] * len(future_years)})

future_predictions = ensemble_model.predict(future_data)

# Plot future temperature trend predictions
plt.figure(figsize=(10, 6))
plt.plot(future_years, future_predictions, marker='o', linestyle='-', color='red')
plt.title('Predicted Temperature Trend for the Next 10 Years')
plt.xlabel('Year')
plt.ylabel('Predicted Temperature (°C)')
plt.show()
