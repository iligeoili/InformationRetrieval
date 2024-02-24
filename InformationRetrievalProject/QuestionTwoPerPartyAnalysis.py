import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from InformationRetrievalProject.GreekStopWords import greek_stop_words  # Import your list of Greek stop words


# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Ομαδοποίηση των ομιλιών ανά κόμμα
grouped = df.groupby('political_party')


def extract_keywords(speeches):
    # Συνδυασμός όλων των ομιλιών ενός βουλευτή σε ένα κείμενο
    combined_speeches = ' '.join(speeches)
    if not combined_speeches.strip():
        return []

    vectorizer = CountVectorizer(stop_words=greek_stop_words)
    word_count_matrix = vectorizer.fit_transform([combined_speeches])
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf_matrix = tfidf.fit_transform(word_count_matrix)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    episode = dense[0].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    top_keywords = [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:10]
    return top_keywords


# Εξαγωγή και εκτύπωση των κυριότερων λέξεων-κλειδιών για κάθε βουλευτή
for name, group in grouped:
    keywords = extract_keywords(group['cleaned_speech'])
    print(f"Κυριότερες λέξεις-κλειδιά για το κόμμα  {name}:")
    for keyword, score in keywords:
        print(f"{keyword}: {score}")
    print("\n")
