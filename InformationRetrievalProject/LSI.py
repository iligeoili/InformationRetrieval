import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE



# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Εξαγωγή διανυσμάτων χαρακτηριστικών για κάθε ομιλία με TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['cleaned_speech'])



# Εφαρμογή LSI
lsi = TruncatedSVD(n_components=100)  # Ο αριθμός των διαστάσεων στον χώρο χαμηλότερων διαστάσεων
lsi_matrix = lsi.fit_transform(tfidf_matrix)

# Απόκτηση των σημαντικότερων θεματικών περιοχών
terms = tfidf.get_feature_names_out()
for i, component in enumerate(lsi.components_):
    terms_component = zip(terms, component)
    sorted_terms = sorted(terms_component, key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:10]
    print(f"Topic {i}:")
    for term in top_terms:
        print(term)
    print("\n")



# Applying t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)  # These parameters can be adjusted
tsne_results = tsne.fit_transform(lsi_matrix)

# Plotting the t-SNE results
plt.figure(figsize=(10, 6))  # Making the plot larger for better visibility
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Greek Parliament Speeches')
plt.show()
#
# # Προαιρετικά: Απεικόνιση των ομιλιών σε 2D χώρο
# plt.scatter(lsi_matrix[:, 0], lsi_matrix[:, 1])
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.title('LSI 2D Representation of Speeches')
# plt.show()

