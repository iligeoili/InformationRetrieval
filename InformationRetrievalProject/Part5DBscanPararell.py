import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import spacy
import multiprocessing

i = 0
# Load the data
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')

# Load the Greek model
nlp = spacy.load('el_core_news_sm')

# Load the Greek stop words using NLTK
nltk.download('stopwords')
greek_stop_words = list(stopwords.words('greek'))

def preprocess_text(document):
    # Text cleaning
    document = re.sub(r'\W', ' ', str(document))
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = re.sub(r'^b\s+', '', document)
    document = document.lower()
    global i
    i += 1
    print(i)
    # Lemmatization
    doc = nlp(document)
    return ' '.join([word.lemma_ for word in doc if word.lemma_ not in greek_stop_words])

# Parallel processing of text preprocessing
def preprocess_documents(documents):
    with multiprocessing.Pool() as pool:
        return pool.map(preprocess_text, documents)

# Apply preprocessing using multiprocessing
df['processed_text'] = preprocess_documents(df['speech'].tolist())

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words=greek_stop_words)
tfidf_matrix = tfidf.fit_transform(df['processed_text'])

# Optional: Scaling the TF-IDF matrix
X_scaled = StandardScaler(with_mean=False).fit_transform(tfidf_matrix)

# K-Means Clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Add cluster information to the DataFrame
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
