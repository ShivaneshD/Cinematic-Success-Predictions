import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Final.xlsx'  # Update the path to your dataset
data = pd.read_excel(file_path)

# Data Cleaning and Preprocessing
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

data['IsBlockbuster'] = data['Verdict'].apply(lambda x: 1 if x == 'Blockbuster' else 0)

# Feature Engineering: Calculate director success rate
director_performance = data.groupby('Directors')['IsBlockbuster'].mean().reset_index()
director_performance.columns = ['Directors', 'DirectorSuccessRate']
data = data.merge(director_performance, on='Directors', how='left')

data = data.join(data['Genres'].str.get_dummies(sep=', '))

# Define features for clustering
features = data.drop(columns=['Original Title', 'Release Date', 'Directors', 'Budget', 'Box office', 'Verdict', 'IsBlockbuster', 'Genres'])
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

# Insights: Identify top directors with more blockbusters in each cluster
for cluster in data['Cluster'].unique():
    if cluster != -1:  # -1 indicates noise
        cluster_data = data[data['Cluster'] == cluster]
        blockbuster_counts = cluster_data[cluster_data['IsBlockbuster'] == 1]['Directors'].value_counts().reset_index()
        blockbuster_counts.columns = ['Directors', 'Blockbuster Count']
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Blockbuster Count', y='Directors', data=blockbuster_counts.head(10), palette='viridis')
        plt.title(f'Top 10 Directors with the Most Blockbusters in Cluster {cluster}')
        plt.xlabel('Blockbuster Count')
        plt.ylabel('Directors')
        plt.show()

        print(f"\nDirectors with the most blockbusters in Cluster {cluster}:")
        print(blockbuster_counts.sort_values(by='Blockbuster Count', ascending=False))


# Save the clustered dataset to a new file
output_path = "clustered_dataset.xlsx"
data.to_excel(output_path, index=False)
print(f"Clustered dataset saved to {output_path}")