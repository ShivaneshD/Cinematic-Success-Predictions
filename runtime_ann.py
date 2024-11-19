import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
movies_df = pd.read_excel(r'C:\\Users\\shiva\\OneDrive\\Desktop\\BDA PROJECT\\Final.xlsx', sheet_name='Sheet1')

# Filter movies with runtime greater than 100 minutes
movies_df = movies_df[movies_df['Runtime (mins)'] > 100]

# Define success criteria
success_criteria = ["Blockbuster", "Super Hit", "Hit"]
movies_df['success'] = movies_df['Verdict'].apply(lambda x: 1 if x in success_criteria else 0)
movies_df['success_category'] = movies_df['Verdict'].apply(lambda x: x if x in success_criteria else 'Other')

# Remove "Other" category
movies_df = movies_df[movies_df['success_category'] != 'Other']

# Handle genres - Convert multiple genres into separate dummy variables
genres_dummies = movies_df['Genres'].str.get_dummies(sep=', ')
movies_df = movies_df.join(genres_dummies)

# Prepare features and target
X = movies_df[['Runtime (mins)'] + genres_dummies.columns.tolist()]
y = movies_df['success']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()

# Input layer (with the shape of X_train) and hidden layers
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

# Output layer (1 neuron for binary classification, using sigmoid activation)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report for detailed metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the training and validation accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize success category vs runtime distribution (Boxplot and Histogram)
fig_box = px.box(movies_df, x='success_category', y='Runtime (mins)', 
                 title='Boxplot of Runtime by Success Category',
                 labels={'Runtime (mins)': 'Runtime (mins)', 'success_category': 'Success Category'})
fig_box.show()

fig_histogram = px.histogram(movies_df, x='Runtime (mins)', color='success_category', nbins=30, 
                             title='Histogram of Runtime by Success Category',
                             labels={'Runtime (mins)': 'Runtime (mins)', 'success_category': 'Success Category'})
fig_histogram.show()
