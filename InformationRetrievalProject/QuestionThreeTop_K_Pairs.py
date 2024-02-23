import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_cleaned.csv')

# Ομαδοποίηση των ομιλιών ανά βουλευτή
grouped = df.groupby('member_name')['cleaned_speech'].apply(' '.join)

# Εξαγωγή διανυσμάτων χαρακτηριστικών με TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(grouped)


# Υπολογισμός ομοιότητας κοσινουσίου
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Μετατροπή του πίνακα ομοιότητας σε DataFrame
similarity_df = pd.DataFrame(cosine_sim, index=grouped.index, columns=grouped.index)

# Εύρεση των top-k ζευγών με τον υψηλότερο βαθμό ομοιότητας
k = 5  # Θέστε το k σύμφωνα με την επιθυμία σας
top_pairs = {}

for i in similarity_df.columns:
    # Ταξινόμηση των ομοιοτήτων σε φθίνουσα σειρά και αγνόηση της πρώτης τιμής (ομοιότητα με τον εαυτό του)
    sorted_indices = np.argsort(similarity_df[i])[::-1][1:]
    sorted_values = similarity_df[i][sorted_indices]
    print(i)
    # Εύρεση των top-k ζευγών
    top_pairs[i] = [(grouped.index[j], sorted_values[j]) for j in sorted_indices[:k]]

# Εμφάνιση των top-k ζευγών
for member, pairs in top_pairs.items():
    print(f"Top-k Similar Pairs for {member}:")
    for pair in pairs:
        print(f"{pair[0]} with similarity {pair[1]:.4f}")
    print("\n")

