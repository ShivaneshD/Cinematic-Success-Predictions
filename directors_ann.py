import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_excel("Final.xlsx")

# Data Cleaning and Preprocessing
data['Budget'] = data['Budget'].str.replace('₹', '').str.replace(' crores', '')
data['Budget'] = data['Budget'].astype(float).fillna(0)

data['Box office'] = data['Box office'].str.replace('₹', '').str.replace(' crores', '')
data['Box office'] = data['Box office'].astype(float).fillna(0)

data['IsBlockbuster'] = data['Verdict'].apply(lambda x: 1 if x == 'Blockbuster' else 0)

# Feature Engineering: Calculate director success rate
director_performance = data.groupby('Directors')['IsBlockbuster'].mean().reset_index()
director_performance.columns = ['Directors', 'DirectorSuccessRate']
data = data.merge(director_performance, on='Directors', how='left')

data = data.join(data['Genres'].str.get_dummies(sep=', '))

# Define features and target variable
features = data.drop(columns=['Original Title', 'Release Date', 'Directors', 'Budget', 'Box office', 'Verdict', 'IsBlockbuster', 'Genres'])
target = data['IsBlockbuster']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(128, input_dim=X_resampled.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

# Train the model
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting the loss and accuracy curves
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label=f'ANN (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
