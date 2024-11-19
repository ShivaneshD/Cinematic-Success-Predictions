#RandomForestClassifier
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Optimize Random Forest Classifier using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report for detailed metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

# Analyze the runtime range for hits, super hits, and blockbusters for each genre
movies_df_melted = movies_df.melt(id_vars=['Runtime (mins)', 'success_category'], value_vars=genres_dummies.columns.tolist(),
                                  var_name='genre', value_name='present')
movies_df_melted = movies_df_melted[movies_df_melted['present'] == 1].drop(columns='present')


# Boxplot
fig_box = px.box(movies_df, x='success_category', y='Runtime (mins)', 
                 title='Boxplot of Runtime by Success Category',
                 labels={'Runtime (mins)': 'Runtime (mins)', 'success_category': 'Success Category'})
fig_box.show()

# Histogram
fig_histogram = px.histogram(movies_df, x='Runtime (mins)', color='success_category', nbins=30, 
                             title='Histogram of Runtime by Success Category',
                             labels={'Runtime (mins)': 'Runtime (mins)', 'success_category': 'Success Category'})
fig_histogram.show()

# Treemap
fig_treemap = px.treemap(movies_df, path=['Genres', 'success_category'], values='success', 
                         title='Treemap of Success by Genre',
                         labels={'success': 'Success'})
fig_treemap.show()
