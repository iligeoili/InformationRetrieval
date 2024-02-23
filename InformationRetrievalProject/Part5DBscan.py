import nltk
from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import  TruncatedSVD
from stanza.pipeline.external import spacy
import spacy
# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')
i = 0
# Φόρτωση του Greek model
nlp = spacy.load('el_core_news_sm')

# Φόρτωση των stop words της ελληνικής γλώσσας μέσω του NLTK
nltk.download('stopwords')
greek_stop_words = list(stopwords.words('greek'))

def preprocess_text(document):
    # Καθαρισμός του κειμένου
    document = re.sub(r'\W', ' ', str(document))
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = re.sub(r'^b\s+', '', document)
    document = document.lower()
    global i
    i += 1
    print(i)
    # Λεμματοποίηση
    document = nlp(document)
    document = [word.lemma_ for word in document if word.lemma_ not in greek_stop_words]

    return ' '.join(document)

# Εφαρμογή της προεπεξεργασίας στο DataFrame
# Υποθέτουμε ότι το DataFrame σας ονομάζεται df και η στήλη με τα κείμενα ονομάζεται 'text'
df['processed_text'] = df['speech'].apply(preprocess_text)
# Φτιάξτε τον TfidfVectorizer με τον πίνακα των stop words
tfidf = TfidfVectorizer(stop_words=greek_stop_words)

# Εφαρμόστε τον TF-IDF Vectorizer στα δεδομένα
tfidf_matrix = tfidf.fit_transform(df['processed_text'])
# Scaling the TF-IDF matrix (optional, but can improve results)
X_scaled = StandardScaler(with_mean=False).fit_transform(tfidf_matrix)

## Number of clusters
k = 5

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)  # Assuming you use the TF-IDF matrix here

# Add cluster information to your DataFrame
df['cluster'] = clusters
df.to_csv('processed_speeches.csv', index=False)
# Visualization with TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=2)
tfidf_matrix_2d = svd.fit_transform(tfidf_matrix)


# Create the scatter plot
plt.scatter(tfidf_matrix_2d[:, 0], tfidf_matrix_2d[:, 1], c=df['cluster'], cmap='rainbow')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('TF-IDF 2D Representation of Speeches with K-Means Clusters')
plt.show()


















































