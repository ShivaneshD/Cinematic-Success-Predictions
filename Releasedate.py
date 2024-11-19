import pandas as pd
import plotly.express as px
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load datasets
movies_df = pd.read_excel(r'C:\\Users\\shiva\\OneDrive\\Desktop\\BDA PROJECT\\Final.xlsx', sheet_name='Sheet1')
festivals_df = pd.read_excel(r'C:\\Users\\shiva\\OneDrive\\Desktop\\BDA PROJECT\\festivals.xlsx', sheet_name='Sheet1')

# Convert date columns to datetime
movies_df['release_date'] = pd.to_datetime(movies_df['Release Date'])
festivals_df['date'] = pd.to_datetime(festivals_df['date'])

# Create a window period around festivals (e.g., 7 days before and after the festival date)
window_days = 2
festivals_df['start_date'] = festivals_df['date'] - timedelta(days=window_days)
festivals_df['end_date'] = festivals_df['date'] + timedelta(days=window_days)

# Join movies and festivals data
joined_df = pd.merge_asof(
    movies_df.sort_values('release_date'),
    festivals_df.sort_values('date'),
    left_on='release_date',
    right_on='date',
    direction='nearest',
    tolerance=pd.Timedelta(days=window_days)
)

# Define success criteria
success_criteria = ["Blockbuster", "Super Hit", "Hit"]
joined_df['success'] = joined_df['Verdict'].apply(lambda x: 1 if x in success_criteria else 0)

# Filter main festivals
main_festivals = [
    "Holi", "Ganesh Chaturthi", "Onam", "Dussehra", "Diwali", "Durga Puja",
    "Independence Day", "Republic Day", "Pongal", "Eid ul Fitr", "Ramzan",
    "Christmas", "Vijaya Dasami", "Tamil New Year", "New Year"
]
filtered_df = joined_df[joined_df['festival'].isin(main_festivals)]

# Feature engineering
X = pd.get_dummies(filtered_df[['festival']], drop_first=True)
y = filtered_df['success']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": accuracy})
    
    # Print detailed metrics
    print(f"\n{name} Model Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot accuracy differences
fig = px.bar(
    results_df,
    x="Model",
    y="Accuracy",
    text="Accuracy",
    title="Accuracy Comparison of Different Models",
    labels={"Accuracy": "Accuracy"},
    hover_name="Model"
)

# Customize the plot
fig.update_layout(
    xaxis_title="Model",
    yaxis_title="Accuracy",
    xaxis_tickangle=-45
)

# Show the plot
fig.show()
