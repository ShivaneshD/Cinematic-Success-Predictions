# xgboost Classifier
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

# Create and train the XGBoost model with hyperparameter tuning
xgb_classifier = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Hyperparameter tuning
xgb_random = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_dist, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
xgb_random.fit(X_train, y_train)

# Best estimator
best_xgb_classifier = xgb_random.best_estimator_

# Make predictions
y_pred = best_xgb_classifier.predict(X_test)
y_pred_proba = best_xgb_classifier.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and display accuracy
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate and display ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")

# Insights: Identify top directors with more blockbusters
blockbuster_counts = data[data['IsBlockbuster'] == 1]['Directors'].value_counts().reset_index()
blockbuster_counts.columns = ['Directors', 'Blockbuster Count']

# Visualize directors with the most blockbusters
plt.figure(figsize=(12, 6))
sns.barplot(x='Blockbuster Count', y='Directors', data=blockbuster_counts.head(10), palette='viridis')
plt.title('Top 10 Directors with the Most Blockbusters')
plt.xlabel('Blockbuster Count')
plt.ylabel('Directors')
plt.show()

print("\nDirectors with the most blockbusters:")
print(blockbuster_counts.sort_values(by='Blockbuster Count', ascending=False))
