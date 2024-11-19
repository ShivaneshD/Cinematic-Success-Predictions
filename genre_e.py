#logistics 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
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
param_grid_lr = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
}

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Use GridSearchCV for hyperparameter tuning for Logistic Regression
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search_lr.fit(X_resampled, y_resampled)

# Get the best parameters for Logistic Regression
best_params_lr = grid_search_lr.best_params_
print("Best Parameters for Logistic Regression:", best_params_lr)
best_lr_model = grid_search_lr.best_estimator_

# Create an ensemble model (VotingClassifier)
ensemble_model = VotingClassifier(estimators=[
    ('lr', best_lr_model),
    ('xgb', xgb_model)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_resampled, y_resampled)

# Predictions and Evaluation
y_pred = ensemble_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report for Ensemble Model:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix for Ensemble Model:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance for Logistic Regression (from ensemble)
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
print("The ensemble model combines the strengths of both Logistic Regression and XGBoost.")
print("It improves the prediction performance by leveraging the diversity of the models.")
