# AQI Prediction using Linear Regression and Random Forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("AQI_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing AQI values
df = df.dropna(subset=['AQI'])

# Fill missing numerical values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

df = df.drop(columns=['City', 'Date', 'AQI_Bucket'])

# Define features and target
X = df.drop('AQI', axis=1)
y = df['AQI']

print("\nFeatures:")
print(X.columns)

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Evaluate Linear Regression
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n===== LINEAR REGRESSION RESULTS =====")
print("MAE      :", round(lr_mae, 2))
print("RMSE     :", round(lr_rmse, 2))
print("R2 Score :", round(lr_r2, 4))

# Train Random Forest model
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

rf.fit(X_train, y_train)

# Predict using Random Forest
rf_pred = rf.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE      :", round(rf_mae, 2))
print("RMSE     :", round(rf_rmse, 2))
print("R2 Score :", round(rf_r2, 4))

# Compare both models
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2 Score": [lr_r2, rf_r2]
})

# Hyperparameter Tuning using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50, None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # Number of random combinations to try
    cv=5,               # 5-Fold Cross Validation
    scoring='r2',
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best Parameters:")
print(random_search.best_params_)

print("\nBest Cross Validation R2 Score:")
print(random_search.best_score_)

# Best Tuned Model
rf = random_search.best_estimator_
print("\n===== MODEL COMPARISON =====")
print(comparison)

# Calculate feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)


plt.title("Feature Importance")
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

# Display sample predictions
sample_predictions = pd.DataFrame({
    "Actual AQI": y_test.iloc[:10].values,
    "Predicted AQI": rf_pred[:10]
})

print("\nSample Predictions:")
print(sample_predictions)

# Save Random Forest model
joblib.dump(rf, "aqi_random_forest_model.pkl")

print("\nModel Saved Successfully!")