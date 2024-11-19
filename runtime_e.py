import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

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

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

# Create an ensemble of RandomForest and XGBoost using VotingClassifier
# Here, we use soft voting for RandomForest and hard voting (predicting directly) for XGBoost
voting_clf = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb_classifier)], voting='hard')

# Fit the ensemble model
voting_clf.fit(X_train, y_train)

# Make predictions with the ensemble model
y_pred = voting_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')

# Print classification report for detailed metrics
print("Classification Report for Ensemble Model:\n", classification_report(y_test, y_pred))

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
