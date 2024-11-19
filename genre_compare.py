import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns

# Load the dataset (update file path as necessary)
file_path = 'Final.xlsx'
data = pd.read_excel(file_path)

# Data Preprocessing
# Convert 'Budget' and 'Box office' columns from strings to numeric values
data['Budget'] = data['Budget'].replace('[₹, crores]', '', regex=True).astype(float)
data['Box office'] = data['Box office'].replace('[₹, crores]', '', regex=True).astype(float)

# Encode 'Verdict' as a binary target variable (1 for 'Blockbuster', 0 otherwise)
data['Is_Blockbuster'] = data['Verdict'].apply(lambda x: 1 if x == 'Blockbuster' else 0)

# Split genres into individual labels and binarize using MultiLabelBinarizer
data['Genres'] = data['Genres'].str.split(', ')
mlb = MultiLabelBinarizer()
genres_dummies = mlb.fit_transform(data['Genres'])
genres_df = pd.DataFrame(genres_dummies, columns=mlb.classes_)

# Combine genres and box office features
features = pd.concat([genres_df], axis=1)
X = features
y = data['Is_Blockbuster']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# ======================================================================
# 1. Logistic Regression Model
# ======================================================================
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_resampled, y_resampled)

# Best Logistic Regression model
best_lr_model = grid_search.best_estimator_

# Predictions and Evaluation for Logistic Regression
y_pred_lr = best_lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# ======================================================================
# 2. ANN Model
# ======================================================================
# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_resampled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN model
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the ANN model
y_pred_proba = model.predict(X_test)
y_pred_ann = (y_pred_proba > 0.5).astype(int)
accuracy_ann = accuracy_score(y_test, y_pred_ann)

# ======================================================================
# Comparison of Results
# ======================================================================
# Print the results for both models
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
print(f"ANN Accuracy: {accuracy_ann * 100:.2f}%")

# Classification Report and Confusion Matrix for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Classification Report and Confusion Matrix for ANN
print("\nANN Classification Report:")
print(classification_report(y_test, y_pred_ann))
print("ANN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ann))

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix - Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap="Blues", ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')

# Confusion Matrix - ANN
sns.heatmap(confusion_matrix(y_test, y_pred_ann), annot=True, fmt='d', cmap="Blues", ax=ax[1])
ax[1].set_title('ANN Confusion Matrix')

plt.show()

# Plot Training and Validation Accuracy for ANN
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ANN Training and Validation Accuracy')
plt.legend()
plt.show()
