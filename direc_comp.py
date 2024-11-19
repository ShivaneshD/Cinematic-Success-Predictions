import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
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

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

# Train and evaluate models
results = []
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-Score": report['1']['f1-score']
    })
    
    # Print model-specific details
    print(f"\n{name} Model Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
plt.title("Accuracy Comparison of Different Models")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.show()

# Display results
print("\nModel Comparison Results:")
print(results_df)
