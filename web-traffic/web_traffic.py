import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
data = pd.read_csv('website_data.csv')  # Adjust path if needed

# Data preprocessing
# Encode categorical variable (Traffic Source)
data = pd.get_dummies(data, columns=['Traffic Source'], drop_first=True)

# Define features (X) and target (y)
X = data.drop('Conversion Rate', axis=1)
y = data['Conversion Rate']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualization 1: Feature Importance Plot
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title('Feature Importance in Random Forest Regressor')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Visualization 2: Actual vs. Predicted Conversion Rate
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Conversion Rate')
plt.xlabel('Actual Conversion Rate (%)')
plt.ylabel('Predicted Conversion Rate (%)')
plt.tight_layout()
plt.show()

# Visualization 3: Data Exploration - Page Views vs. Conversion Rate
# Reload original data for visualization (before scaling)
data_orig = pd.read_csv('website_data.csv')
plt.figure(figsize=(8, 6))
sns.boxplot(x=pd.cut(data_orig['Conversion Rate'], bins=5), y='Page Views', data=data_orig, palette='Set2')
plt.title('Page Views vs. Conversion Rate (Binned)')
plt.xlabel('Conversion Rate Bins (%)')
plt.ylabel('Page Views')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()