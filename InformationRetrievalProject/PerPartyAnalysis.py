import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords


#nltk.download('stopwords')
#greek_stop_words = stopwords.words('greek')

# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')

# Ομαδοποίηση των ομιλιών ανά βουλευτή
grouped = df.groupby('member_name')


def extract_keywords(speeches):
    # Συνδυασμός όλων των ομιλιών ενός βουλευτή σε ένα κείμενο
    combined_speeches = ' '.join(speeches)

    greek_stop_words = [
        'και', 'στο', 'στη', 'με', 'για', 'αλλά', 'ή', 'σαν', 'είναι',
        'στην', 'στα', 'την', 'του', 'το', 'της', 'των', 'τον', 'που',
        'τον', 'την', 'το', 'τα', 'τους', 'τις', 'μια', 'μία', 'ένα', 'ένας',
        'μετά', 'πριν', 'πάνω', 'μέσα', 'κάτω', 'από', 'πάνω', 'όπως',
        'ότι', 'όταν', 'επειδή', 'εάν', 'αφού', 'προτού', 'ενώ', 'αν',
        'ήταν', 'έχει', 'είχε', 'είχαν', 'έχουν', 'έχω', 'έχετε', 'είμαι',
        'είσαι', 'είναι', 'είμαστε', 'είστε', 'στον', 'να', 'δε', 'δεν',
        'μη', 'μην', 'είτε', 'ούτε', 'αλλιώς', 'παρά', 'έτσι', 'όσο', 'σας', 'μου',
        'μας', 'σε', 'τι', 'οι', 'αυτά', 'ως', 'τι', 'τη', 'στις', 'αυτό', 'αυτή', 'θα', 'εμείς', 'οποία',
        'όμως', 'οποίο', 'πολύ', 'δηλαδή', 'κυρίες', 'κύριε', 'κατά', 'λοιπόν', 'κύριοι', 'αυτήν',
        'διότι', 'γιατί', 'προς', 'πω', 'κάθε', 'πιο', 'γι', '000'

    ]

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
    keywords = extract_keywords(group['speech'])
    print(f"Κυριότερες λέξεις-κλειδιά για {name}:")
    for keyword, score in keywords:
        print(f"{keyword}: {score}")
    print("\n")
