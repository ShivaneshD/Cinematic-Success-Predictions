import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import seaborn as sns

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

# Check class distribution in y_train
class_counts = y_train.value_counts()
print("Class distribution in y_train:")
print(class_counts)

# Apply SMOTE to balance the classes in the training set if there are more than one class
if len(class_counts) > 1:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    print("Only one class present in y_train, SMOTE will not be applied.")
    X_train_resampled, y_train_resampled = X_train, y_train

# Random Forest Classifier Model
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')  # Add class_weight='balanced' to handle imbalance
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_rf = grid_search.best_estimator_

# Standardize the features for ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
ann_model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
ann_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN model
ann_model.fit(X_train_scaled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the models

# Random Forest Evaluation
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

# Display Random Forest results
print("\nRandom Forest Model:")
print(f"Accuracy: {accuracy_rf:.4f}")
print("Classification Report:")
print(report_rf)

# ANN Evaluation
y_pred_ann = (ann_model.predict(X_test_scaled) > 0.5).astype(int)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
report_ann = classification_report(y_test, y_pred_ann)

# Display ANN results
print("\nANN Model:")
print(f"Accuracy: {accuracy_ann:.4f}")
print("Classification Report:")
print(report_ann)

# Calculate ROC AUC scores for both models, ensuring both classes are present
try:
    # Ensure we are getting probabilities for both classes for Random Forest
    if len(set(y_test)) > 1:
        roc_auc_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
    else:
        roc_auc_rf = None
except ValueError:
    print("Skipping ROC AUC calculation for Random Forest due to only one class in y_test")
    roc_auc_rf = None

try:
    # Ensure we are getting probabilities for both classes for ANN
    if len(set(y_test)) > 1:
        roc_auc_ann = roc_auc_score(y_test, ann_model.predict(X_test_scaled))
    else:
        roc_auc_ann = None
except ValueError:
    print("Skipping ROC AUC calculation for ANN due to only one class in y_test")
    roc_auc_ann = None

# Prepare comparison data (Adding more metrics)
metrics = ['precision', 'recall', 'f1-score']
comparison_data = {
    'Metric': ['Accuracy'] + [f"{metric} (Class {i})" for i in [0, 1] for metric in metrics],
    'Random Forest': [
        accuracy_rf
    ] + [
        report_rf.get(str(i), {}).get(metric, 'N/A') for i in [0, 1] for metric in metrics
    ],
    'ANN': [
        accuracy_ann
    ] + [
        report_ann.get(str(i), {}).get(metric, 'N/A') for i in [0, 1] for metric in metrics
    ]
}

# Add AUC Scores to the comparison
comparison_data['Metric'].append('ROC AUC')
comparison_data['Random Forest'].append(roc_auc_rf if roc_auc_rf is not None else 'N/A')
comparison_data['ANN'].append(roc_auc_ann if roc_auc_ann is not None else 'N/A')

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_data)

# Print the comparison table
print("\nModel Comparison Table:")
print(comparison_df)

# Save the results as CSV
comparison_df.to_csv("model_comparison.csv", index=False)

# Plot ROC Curve for both models if AUC scores are available
if roc_auc_rf is not None and roc_auc_ann is not None:
    fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
    fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_model.predict(X_test_scaled))

    plt.figure(figsize=(12, 8))
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {roc_auc_ann:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

# Plot Confusion Matrices for both models
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
ax[0].set_title("Random Forest Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_ann), annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False)
ax[1].set_title("ANN Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.show()
