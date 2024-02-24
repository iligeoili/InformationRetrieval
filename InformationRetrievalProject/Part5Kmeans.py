from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

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

# Apply PCA to reduce the dimensionality of the TF-IDF matrix to 2 dimensions
#pca = PCA(n_components=2)
#tfidf_matrix_2d = pca.fit_transform(tfidf_matrix.toarray())

# Use TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=2)
tfidf_matrix_2d = svd.fit_transform(tfidf_matrix)

# Create the scatter plot
plt.scatter(tfidf_matrix_2d[:, 0], tfidf_matrix_2d[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('TF-IDF 2D Representation of Speeches with K-Means Clusters')
plt.show()
