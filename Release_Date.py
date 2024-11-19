#LogisticRegression
import pandas as pd
import plotly.express as px
from datetime import timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load movies dataset using Pandas
movies_df = pd.read_excel(r'C:\\Users\\shiva\\OneDrive\\Desktop\\BDA PROJECT\\Final.xlsx', sheet_name='Sheet1')

# Load festivals dataset using Pandas
festivals_df = pd.read_excel(r'C:\\Users\\shiva\\OneDrive\\Desktop\\BDA PROJECT\\festivals.xlsx', sheet_name='Sheet1')

# Convert date columns to datetime
movies_df['release_date'] = pd.to_datetime(movies_df['Release Date'])
festivals_df['date'] = pd.to_datetime(festivals_df['date'])

# Create a window period around festivals (e.g., 7 days before and after the festival date)
window_days = 2
festivals_df['start_date'] = festivals_df['date'] - timedelta(days=window_days)
festivals_df['end_date'] = festivals_df['date'] + timedelta(days=window_days)

# Join movies and festivals data on the release date within the festival window period
joined_df = pd.merge_asof(movies_df.sort_values('release_date'), 
                          festivals_df.sort_values('date'), 
                          left_on='release_date', 
                          right_on='date', 
                          direction='nearest', 
                          tolerance=pd.Timedelta(days=window_days))

# Define success criteria
success_criteria = ["Blockbuster", "Super Hit", "Hit"]
joined_df['success'] = joined_df['Verdict'].apply(lambda x: 1 if x in success_criteria else 0)

# Main festivals to display in the chart
main_festivals = [
    "Holi", "Ganesh Chaturthi", "Onam", "Dussehra","Diwali", "Durga Puja", 
    "Independence Day", "Republic Day", "Pongal", "Eid ul Fitr", "Ramzan", 
    "Christmas", "Vijaya Dasami", "Tamil New Year", "New Year"
]

# Filter the main festivals
filtered_df = joined_df[joined_df['festival'].isin(main_festivals)]

# Feature engineering
X = pd.get_dummies(filtered_df[['festival']], drop_first=True)
y = filtered_df['success']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type (L1 or L2)
    'solver': ['liblinear', 'saga']  # Solvers for L1 and L2
}
lr = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_lr = grid_search.best_estimator_

# Evaluate the model
y_pred = best_lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report for detailed metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

# Analyze the success rate for each festival
success_rate_df = filtered_df.groupby('festival').agg(blockbusters=('success', 'sum'), 
                                                      total_movies=('Original Title', 'count')).reset_index()
success_rate_df['success_rate'] = success_rate_df['blockbusters'] / success_rate_df['total_movies']

# Plot the success rate with Plotly and display the count of movies above the bars
fig = px.bar(success_rate_df, x='festival', y='success_rate', text='total_movies', title='Success Rate of Movies Released Around Main Festivals',
             labels={'success_rate': 'Success Rate'}, hover_name='festival')

# Customize the layout for better visualization
fig.update_layout(
    xaxis_title="Festival",
    yaxis_title="Success Rate",
    xaxis_tickangle=-45,
    hovermode="x unified"
)

# Show the plot
fig.show()
