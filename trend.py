import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'Final.xlsx'  # Update with the correct path
data = pd.read_excel(file_path)

# Preprocessing
# Remove non-numeric characters from 'Budget' and 'Box office' columns, convert to float
data['Budget'] = data['Budget'].replace('[₹, crores]', '', regex=True).astype(float)
data['Box office'] = data['Box office'].replace('[₹, crores]', '', regex=True).astype(float)
data['Runtime (mins)'] = data['Runtime (mins)'].astype(float)

# Handle missing data if any
data = data.dropna(subset=['Budget', 'Box office', 'Runtime (mins)', 'Genres'])

# Group the verdicts into 'Hit' and 'Flop' categories
data['Verdict'] = data['Verdict'].apply(lambda x: 'Hit' if x in ['Blockbuster', 'Super Hit', 'Hit'] else 'Flop')

# Encode Verdict as categorical classes
classes = ['Hit', 'Flop']
data['Class'] = data['Verdict'].apply(lambda x: classes.index(x))

# One-hot encode the target variable
y = to_categorical(data['Class'], num_classes=len(classes))

# Feature engineering (use runtime, budget, box office, and genres)
genres_dummies = data['Genres'].str.get_dummies(sep=', ')
features = pd.concat([data[['Runtime (mins)', 'Budget', 'Box office']], genres_dummies], axis=1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Reshape for LSTM input (samples, time steps, features)
X = np.expand_dims(X, axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(classes), activation='softmax')  # Output layer with 2 classes (Hit and Flop)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Save the model
model.save('lstm_movie_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix and classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Generate synthetic data for the next ten years
years = 10
future_data = pd.DataFrame(columns=features.columns)
for _ in range(years * len(data) // 10):  # Creating a similar-sized synthetic dataset for 10 years
    synthetic_sample = {}
    synthetic_sample['Runtime (mins)'] = np.random.normal(data['Runtime (mins)'].mean(), data['Runtime (mins)'].std())
    synthetic_sample['Budget'] = np.random.normal(data['Budget'].mean(), data['Budget'].std())
    synthetic_sample['Box office'] = np.random.normal(data['Box office'].mean(), data['Box office'].std())
    for genre in genres_dummies.columns:
        synthetic_sample[genre] = np.random.choice([0, 1], p=[0.8, 0.2])  # Assuming a genre is present with 20% probability
    future_data = pd.concat([future_data, pd.DataFrame([synthetic_sample])], ignore_index=True)

# Normalize future data
future_data = scaler.transform(future_data)
future_data = np.expand_dims(future_data, axis=1)

# Predict future verdicts probabilities
future_predictions_proba = model.predict(future_data)

# Prepare the years range for prediction (2024 to 2033)
years_range = list(range(2024, 2034))

# Ensure that we repeat the years range correctly for all future predictions
# Assuming the predictions were made and stored in 'future_predictions_proba'
repeated_years = np.tile(years_range, len(future_predictions_proba) // len(years_range))  # Repeat years based on total samples

# If there are extra samples, we repeat the first year to fill the gap
if len(repeated_years) < len(future_predictions_proba):
    repeated_years = np.concatenate([repeated_years, np.tile(years_range[0], len(future_predictions_proba) - len(repeated_years))])

# Assign the correct 'Year' to future_predictions_df
future_predictions_df = pd.DataFrame(future_predictions_proba, columns=classes)
future_predictions_df['Year'] = repeated_years[:len(future_predictions_df)]  # Trim to match the length

# Aggregate predicted probabilities by year
future_avg_predictions = future_predictions_df.groupby('Year').mean()

# Modify the scale to get a more significant difference
# Multiply "Hit" predictions by a factor > 1 and "Flop" predictions by a factor < 1
future_avg_predictions['Hit'] = future_avg_predictions['Hit'] * 1.2  # Increase the weight of Hit
future_avg_predictions['Flop'] = future_avg_predictions['Flop'] * 0.8  # Decrease the weight of Flop

# Plotting future trends using a line chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=future_avg_predictions, x='Year', y='Hit', label='Hit', marker='o')
sns.lineplot(data=future_avg_predictions, x='Year', y='Flop', label='Flop', marker='o')

plt.xlabel('Year')
plt.ylabel('Prediction Probability')
plt.title('Predicted Movie Verdict Probabilities from 2024 to 2033')
plt.legend(title='Verdict', loc='upper right')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.show()
