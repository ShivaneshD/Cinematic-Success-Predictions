import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

# Prepare features for clustering
features = pd.concat([movies_df[['Runtime (mins)']], genres_dummies], axis=1)
features.fillna(0, inplace=True)  # Handle any remaining NaN values

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply OPTICS Clustering
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
clusters = optics.fit_predict(features_scaled)

# Add cluster labels to the dataset
movies_df['Cluster'] = clusters

# Evaluate Clustering
ari = adjusted_rand_score(movies_df['success'], movies_df['Cluster'])
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
    hue=movies_df['Cluster'],
    palette='viridis',
    s=50,
    legend='full'
)
plt.title("OPTICS Clustering Results (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc="best")
plt.show()

# Boxplot for Runtime by Cluster
fig_box = px.box(movies_df, x='Cluster', y='Runtime (mins)', 
                 title='Boxplot of Runtime by Cluster',
                 labels={'Runtime (mins)': 'Runtime (mins)', 'Cluster': 'Cluster'})
fig_box.show()

# Histogram for Runtime by Cluster
fig_histogram = px.histogram(movies_df, x='Runtime (mins)', color='Cluster', nbins=30, 
                             title='Histogram of Runtime by Cluster',
                             labels={'Runtime (mins)': 'Runtime (mins)', 'Cluster': 'Cluster'})
fig_histogram.show()

# Treemap for Success by Genre and Cluster
fig_treemap = px.treemap(movies_df, path=['Genres', 'success_category'], values='success', 
                         title='Treemap of Success by Genre and Cluster',
                         labels={'success': 'Success'})
fig_treemap.show()

# Save the clustered dataset to a new file
output_path = "clustered_dataset_optics.xlsx"
movies_df.to_excel(output_path, index=False)
print(f"Clustered dataset saved to {output_path}")
