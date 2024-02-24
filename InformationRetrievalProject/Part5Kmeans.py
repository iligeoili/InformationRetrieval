from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')


# Φτιάξτε τον TfidfVectorizer με τον πίνακα των stop words
tfidf = TfidfVectorizer()

# Εφαρμόστε τον TF-IDF Vectorizer στα δεδομένα
tfidf_matrix = tfidf.fit_transform(df['cleaned_speech'])

# Εφαρμογή K-Means
k = 10  # Ο αριθμός των ομάδων (μπορείτε να αλλάξετε ανάλογα)
kmeans = KMeans(n_clusters=k)
kmeans.fit(tfidf_matrix)
df['cluster'] = kmeans.labels_

# Εμφάνιση των αποτελεσμάτων του K-Means
print("Results of K-Means Clustering:")
for cluster in range(k):
    cluster_df = df[df['cluster'] == cluster]
    print(f"Cluster {cluster + 1} - Number of Speeches: {len(cluster_df)}")



# Use TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=2)
tfidf_matrix_2d = svd.fit_transform(tfidf_matrix)

# # Create the scatter plot
plt.scatter(tfidf_matrix_2d[:, 0], tfidf_matrix_2d[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('TF-IDF 2D Representation of Speeches with K-Means Clusters')

# Save the plot to a file
plt.savefig('kmeans_clusters_svd.png')

# After saving, you can optionally clear the figure to free memory if you're making more plots
plt.clf()




# Initialize a new figure for 3D plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Extract the three dimensions
# x = tfidf_matrix_2d[:, 0]
# y = tfidf_matrix_2d[:, 1]
# z = tfidf_matrix_2d[:, 2]
#
# # Scatter plot using the first three components from SVD
# scatter = ax.scatter(x, y, z, c=kmeans.labels_, cmap='rainbow', marker='o')
#
# # Label the axes
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
#
# # Title of the plot
# ax.set_title('3D TF-IDF Representation of Speeches with K-Means Clusters')
#
# # Optional: Add a legend
# # Create a custom legend for cluster labels
# # legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
# # ax.add_artist(legend1)
#
# # Save the plot to a file
# plt.savefig('kmeans_clusters_3d_svd.png')
#
# # Show the plot
# plt.show()
#
# # Clear the figure to free memory if you're making more plots
# plt.clf()

# Initialize a new figure for 3D plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Extract the first three dimensions for the coordinates
# x = tfidf_matrix_2d[:, 0]
# y = tfidf_matrix_2d[:, 1]
# z = tfidf_matrix_2d[:, 2]
#
# # Use the fourth dimension to determine the size of the points.
# # You might need to scale the fourth dimension to suitable sizes for plotting.
# # This is a simple scaling example; adjust it according to your data's needs.
# w = tfidf_matrix_2d[:, 3] * 40  # Scale factor for size, adjust as needed
#
# # Scatter plot using the first three components from SVD and size from the fourth component
# scatter = ax.scatter(x, y, z, c=kmeans.labels_, cmap='rainbow', s=w, marker='o')
#
# # Label the axes
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
#
# # Title of the plot
# ax.set_title('4D TF-IDF Representation of Speeches with K-Means Clusters')
#
# plt.savefig('kmeans_clusters_4d_svd.png', dpi=300)  # dpi parameter is optional for resolution
# # Show the plot
# plt.show()
#
# # Clear the figure to free memory if you're making more plots
# plt.clf()