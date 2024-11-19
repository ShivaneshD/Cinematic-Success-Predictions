import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
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

# Define the Logistic Regression model with balanced class weights
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_resampled, y_resampled)

# Get the best parameters and train the model
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_lr_model = grid_search.best_estimator_

# Predictions and Evaluation
y_pred = best_lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
coefficients = best_lr_model.coef_[0]
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': abs(coefficients)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis", hue=None)
plt.xlabel("Feature Importance (Normalized)")
plt.ylabel("Features")
plt.title("Feature Importance for Predicting Blockbusters (Genre and Box Office)")
plt.show()

print("\nInsights:")
print("Logistic Regression now handles class imbalance and provides better predictions for blockbusters.")
