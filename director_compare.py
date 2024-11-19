import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_excel("Final.xlsx")

# Data Cleaning and Preprocessing
data['Budget'] = data['Budget'].str.replace('₹', '').str.replace(' crores', '').astype(float).fillna(0)
data['Box office'] = data['Box office'].str.replace('₹', '').str.replace(' crores', '').astype(float).fillna(0)
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

# Standardize the features for ANN and XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
y_pred_proba_xgb = xgb_classifier.predict_proba(X_test)[:, 1]

# ANN Model
# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Build the ANN model
ann_model = Sequential()
ann_model.add(Dense(128, input_dim=X_resampled.shape[1], activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(64, activation='relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

# Compile and train the ANN model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
ann_model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Predict using ANN
y_pred_proba_ann = ann_model.predict(X_test_scaled)
y_pred_ann = (y_pred_proba_ann > 0.5).astype(int)

# Evaluate XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

# Evaluate ANN
accuracy_ann = accuracy_score(y_test, y_pred_ann)
roc_auc_ann = roc_auc_score(y_test, y_pred_proba_ann)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)
class_report_ann = classification_report(y_test, y_pred_ann)

# Print Comparison Results
print("XGBoost Model Performance:")
print(f"Accuracy: {accuracy_xgb * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc_xgb:.2f}")
print("Confusion Matrix:")
print(conf_matrix_xgb)
print("Classification Report:")
print(class_report_xgb)

print("\nANN Model Performance:")
print(f"Accuracy: {accuracy_ann * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc_ann:.2f}")
print("Confusion Matrix:")
print(conf_matrix_ann)
print("Classification Report:")
print(class_report_ann)

# Plot ROC Curve for both models
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_proba_ann)

plt.figure(figsize=(12, 8))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {roc_auc_ann:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# Plot Confusion Matrices for both models
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
ax[0].set_title("XGBoost Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_ann, annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False)
ax[1].set_title("ANN Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.show()
