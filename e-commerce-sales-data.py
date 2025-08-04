import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the dataset
df = pd.read_csv('E-commerce-Dataset.csv')

# Convert Order_Date to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Aggregate Sales by Order_Date (sum of Sales per day)
df_daily = df.groupby('Order_Date')['Sales'].sum().reset_index()

# Sort by date to ensure chronological order
df_daily = df_daily.sort_values('Order_Date')

# Check for missing dates and fill if necessary
date_range = pd.date_range(start=df_daily['Order_Date'].min(), end=df_daily['Order_Date'].max(), freq='D')
df_daily = df_daily.set_index('Order_Date').reindex(date_range, fill_value=0).reset_index()
df_daily.columns = ['Order_Date', 'Sales']

# Apply 7-day moving average smoothing
df_daily['Sales_Smoothed'] = df_daily['Sales'].rolling(window=10, min_periods=1, center=True).mean()

# Prepare data for LSTM (use smoothed Sales)
sales = df_daily['Sales_Smoothed'].values.reshape(-1, 1)

# Normalize the smoothed Sales data
scaler = MinMaxScaler(feature_range=(0, 1))
sales_scaled = scaler.fit_transform(sales)

# Create sequences for LSTM (use past 10 days to predict the next day)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(sales_scaled, seq_length)

# Split into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, 1), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Prepare data for animated line plot
dates = df_daily['Order_Date'].iloc[-len(y_test):].values
actual_sales = y_test_inv.flatten()
predicted_sales = y_pred_inv.flatten()

# Create animated line plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('E-Commerce Data : Actual vs Predicted Sales')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.grid(True)

# Initialize empty line plots with new colors
line_actual, = ax.plot([], [], '-', color='#8B5CF6', label='Actual Sales', linewidth=2)  # Purple
line_pred, = ax.plot([], [], '-', color='#F59E0B', label='Predicted Sales', linewidth=2)  # Orange
ax.legend()

# Set axis limits
ax.set_xlim(dates[0], dates[-1])
ax.set_ylim(min(min(actual_sales), min(predicted_sales)) * 0.95, max(max(actual_sales), max(predicted_sales)) * 1.05)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Initialize text annotations for sales amounts
text_actual = ax.text(dates[0], min(actual_sales) * 0.95, '', fontsize=10, color='#8B5CF6')
text_pred = ax.text(dates[0], min(predicted_sales) * 0.95, '', fontsize=10, color='#F59E0B')

# Animation update function
def update(frame):
    line_actual.set_data(dates[:frame+1], actual_sales[:frame+1])
    line_pred.set_data(dates[:frame+1], predicted_sales[:frame+1])
    
    # Update text annotations with the latest sales amount
    if frame > 0:
        text_actual.set_x(dates[frame])
        text_actual.set_y(actual_sales[frame])
        text_actual.set_text(f'${actual_sales[frame]:.2f}')
        
        text_pred.set_x(dates[frame])
        text_pred.set_y(predicted_sales[frame])
        text_pred.set_text(f'${predicted_sales[frame]:.2f}')
    return line_actual, line_pred, text_actual, text_pred

# Create animation
ani = FuncAnimation(fig, update, frames=len(dates), interval=100, blit=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Try to save the animation as MP4, fallback to GIF if ffmpeg is unavailable
try:
    ani.save('e-commerce-sales-data.mp4', writer='ffmpeg', fps=10)
    print("Animation saved as 'e-commerce-sales-data.mp4'")
except ValueError as e:
    print(f"Error saving MP4: {e}")
    print("Falling back to saving as GIF...")
    ani.save('e-commerce-sales-data.gif', writer='pillow', fps=10)
    print("Animation saved as 'e-commerce-sales-data.gif'")

# Show the plot
plt.show()

# Visualize training loss
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()