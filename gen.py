import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Final.xlsx'
data = pd.read_excel(file_path)

# Data Preprocessing
data['Budget'] = data['Budget'].replace('[₹, crores]', '', regex=True).astype(float)
data['Box office'] = data['Box office'].replace('[₹, crores]', '', regex=True).astype(float)
data['Is_Blockbuster'] = data['Verdict'].apply(lambda x: 1 if x == 'Blockbuster' else 0)
data['Genres'] = data['Genres'].str.split(', ')

mlb = MultiLabelBinarizer()
genres_dummies = mlb.fit_transform(data['Genres'])
genres_df = pd.DataFrame(genres_dummies, columns=mlb.classes_)

# Combine genres and other features
features = pd.concat([genres_df], axis=1)
X = features
y = data['Is_Blockbuster']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Hyperparameter grids for tuning
param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    "Decision Tree": {"max_depth": [10, 20, None], "criterion": ["gini", "entropy"]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
}

# Model evaluation
results = []
best_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
    grid_search.fit(X_resampled, y_resampled)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((model_name, acc))
    
    print(f"{model_name} Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Compare model performance
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

# Plot comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.xlabel("Accuracy (%)")
plt.ylabel("Models")
plt.title("Model Comparison for Predicting Blockbusters")
plt.show()

# Insights
print("\nInsights:")
print("The best-performing model can be identified from the chart above, and its metrics suggest the strongest predictive capabilities for blockbusters.")
