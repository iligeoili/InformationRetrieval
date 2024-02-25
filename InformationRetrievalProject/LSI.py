import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Εξαγωγή διανυσμάτων χαρακτηριστικών για κάθε ομιλία με TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['cleaned_speech'])

# Εφαρμογή LSI
n_components = 10  # Ο αριθμός των θεματικών περιοχών που θέλετε να εντοπίσετε
lsi = TruncatedSVD(n_components=n_components)
lsi_matrix = lsi.fit_transform(tfidf_matrix)

# Εκτύπωση των σημαντικότερων λέξεων για κάθε θεματική περιοχή
terms = tfidf.get_feature_names_out()
for i, component in enumerate(lsi.components_):
    terms_component = zip(terms, component)
    sorted_terms = sorted(terms_component, key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:10]
    print(f"Κυριότερες Λέξεις της θεματικής ενότητας  {i}:")
    for term in top_terms:
        print(term)
    print("\n")

