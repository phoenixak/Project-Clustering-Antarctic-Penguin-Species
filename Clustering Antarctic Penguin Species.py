# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Examine the Dataset
penguins_df = pd.read_csv("data/penguins.csv")
print(penguins_df.head()) # Display the first few rows of the dataset
print(penguins_df.info()) # Display information about the dataset

# Step 2: Handle Null Values and Outliers
# Visualize boxplots to identify potential outliers
penguins_df.boxplot()  
plt.show()

# Drop rows with null values
penguins_clean = penguins_df.dropna()

# Manually identify and remove outliers based on visual inspection
# Assuming rows 9 and 14 are identified as outliers based on the boxplot
penguins_clean = penguins_clean.drop([9, 14])

# Step 3: Preprocess the Dataset to Create Dummy Variables
# Convert the 'sex' column into dummy variables and drop the original column
df = pd.get_dummies(penguins_clean).drop('sex_.', axis=1)

# Step 4: Perform Preprocessing Steps on the Dataset - Scaling
# Standardize the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Create a DataFrame from the scaled data, retaining the column names
penguins_preprocessed = pd.DataFrame(data=X, columns=df.columns)
print(penguins_preprocessed.head(10)) # Display the first few rows of the preprocessed data

# Step 5: Perform PCA
# Perform PCA to reduce the dimensionality of the dataset
pca = PCA(n_components=None)
pca.fit(penguins_preprocessed)

# Determine the number of components with more than 10% explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
n_components = sum(explained_variance_ratio > 0.1)

# Perform PCA with the optimal number of components
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

# Step 6: Detect the Optimal Number of Clusters for k-means Clustering
# Use the Elbow Method to determine the optimal number of clusters
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)    
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Assuming 4 is the optimal number based on the elbow plot
n_clusters = 4

# Step 7: Run the k-means Clustering Algorithm
# Run k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_PCA)

# Visualize the resulting clusters
plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (K={n_clusters})')
plt.legend()
plt.show()

# Step 8: Create a Final Statistical DataFrame for Each Cluster
# Add the labels from k-means clustering to the original DataFrame
penguins_clean['label'] = kmeans.labels_

# Create a final characteristic DataFrame for each cluster
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'label']
stat_penguins = penguins_clean[numeric_columns].groupby('label').mean()
print(stat_penguins) # Display the statistical DataFrame for each cluster