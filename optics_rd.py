import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import timedelta

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

# Data Preprocessing
data = filtered_df.copy()

# Convert 'Budget' and 'Box office' columns from strings to numeric values
def convert_currency(value):
    if isinstance(value, str):
        value = value.replace('₹', '').replace(' crores', '').replace(' ', '')
        try:
            return float(value)
        except ValueError:
            return 0.0
    return value

data['Budget'] = data['Budget'].apply(convert_currency).fillna(0)
data['Box office'] = data['Box office'].apply(convert_currency).fillna(0)

# Encode 'Verdict' as a binary target variable (1 for 'Blockbuster', 0 otherwise)
data['IsBlockbuster'] = data['Verdict'].apply(lambda x: 1 if x == 'Blockbuster' else 0)

# Feature Engineering: Calculate director success rate
director_performance = data.groupby('Directors')['IsBlockbuster'].mean().reset_index()
director_performance.columns = ['Directors', 'DirectorSuccessRate']
data = data.merge(director_performance, on='Directors', how='left')

# Split genres into individual labels and binarize using MultiLabelBinarizer
data['Genres'] = data['Genres'].str.split(', ')
mlb = MultiLabelBinarizer()
genres_dummies = mlb.fit_transform(data['Genres'])
genres_df = pd.DataFrame(genres_dummies, columns=mlb.classes_)

# Combine features for clustering
features = pd.concat([data[['Runtime (mins)', 'Budget', 'Box office']], genres_df], axis=1)
features.fillna(0, inplace=True)  # Handle any remaining NaN values

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply OPTICS Clustering
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
clusters = optics.fit_predict(features_scaled)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Evaluate Clustering
ari = adjusted_rand_score(data['IsBlockbuster'], data['Cluster'])
silhouette_avg = silhouette_score(features_scaled, clusters)

print(f"Adjusted Rand Index: {ari:.2f}")
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualize Clusters (PCA for dimensionality reduction)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=features_pca[:, 0],
    y=features_pca[:, 1],
    hue=data['Cluster'],
    palette='viridis',
    s=50,
    legend='full'
)
plt.title("OPTICS Clustering Results (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc="best")
plt.show()

# Save the clustered dataset to a new file
output_path = "clustered_dataset.xlsx"
data.to_excel(output_path, index=False)
print(f"Clustered dataset saved to {output_path}")

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
