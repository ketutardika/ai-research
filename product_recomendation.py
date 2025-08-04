import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('ecommerce_orders.csv')

# Data quality check
data['Delivery Time'] = (pd.to_datetime(data['Delivery Date']) - pd.to_datetime(data['Order Date'])).dt.days
data['Shipping Time'] = (pd.to_datetime(data['Shipping Date']) - pd.to_datetime(data['Order Date'])).dt.days
# Check for negative or extreme delivery times
print("Data Quality Check:")
print(data[['Delivery Time', 'Shipping Time']].describe())
if (data['Delivery Time'] < 0).any() or (data['Shipping Time'] < 0).any():
    print("Warning: Negative Delivery Time or Shipping Time detected!")
    data = data[data['Delivery Time'] >= 0]  # Remove invalid rows

# Feature engineering
data['Order Month'] = pd.to_datetime(data['Order Date']).dt.month
data['Order DayOfWeek'] = pd.to_datetime(data['Order Date']).dt.dayofweek
data['Is Delayed'] = data['Delivery Status'].map({'Delivered': 0, 'Delayed': 1})

# Select features and target
features = ['Vendor', 'Product Category', 'Region', 'Shipping Time', 'Order Month', 'Order DayOfWeek', 'Is Delayed']
X = data[features]
y = data['Delivery Time']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Vendor', 'Product Category', 'Region'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Absolute Error (MAE): {mae:.2f} days')
print(f'Mean Squared Error (MSE): {mse:.2f} days²')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f} days')
print(f'R² Score: {r2:.2f}')

# Cross-validation scores
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f'Cross-Validation R² Scores: {cv_scores}')
print(f'Average CV R²: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Visualization 1: Bar Chart of Average Delivery Time by Vendor
plt.figure(figsize=(8, 5))
avg_delivery_time = data.groupby('Vendor')['Delivery Time'].mean().sort_values()
sns.barplot(x=avg_delivery_time.index, y=avg_delivery_time.values, palette='viridis')
plt.title('Average Delivery Time by Vendor')
plt.xlabel('Vendor')
plt.ylabel('Average Delivery Time (Days)')
plt.tight_layout()
plt.show()

# Visualization 2: Box Plot of Delivery Time by Vendor
plt.figure(figsize=(8, 5))
sns.boxplot(x='Vendor', y='Delivery Time', data=data, palette='muted')
plt.title('Delivery Time Distribution by Vendor')
plt.xlabel('Vendor')
plt.ylabel('Delivery Time (Days)')
plt.tight_layout()
plt.show()

# Visualization 3: Scatter Plot of Actual vs Predicted Delivery Time
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Delivery Time (Days)')
plt.ylabel('Predicted Delivery Time (Days)')
plt.title('Actual vs Predicted Delivery Time')
plt.tight_layout()
plt.show()

# Feature Importance
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance.head(10))

# Save plots (optional)
plt.savefig('delivery_time_plots.png')