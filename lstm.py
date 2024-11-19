import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = 'Final.xlsx'  # Update with the correct path
data = pd.read_excel(file_path)

# Preprocessing
data['Budget'] = data['Budget'].replace('[₹, crores]', '', regex=True).astype(float)
data['Box office'] = data['Box office'].replace('[₹, crores]', '', regex=True).astype(float)
data['Runtime (mins)'] = data['Runtime (mins)'].astype(float)

# Encode Verdict as categorical classes
classes = ['Blockbuster', 'Super Hit', 'Hit', 'Flop']
data['Verdict'] = data['Verdict'].apply(lambda x: x if x in classes else 'Flop')
data['Class'] = data['Verdict'].apply(lambda x: classes.index(x))

# One-hot encode the target
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
    Dense(len(classes), activation='softmax')  # Output layer with 4 classes
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
