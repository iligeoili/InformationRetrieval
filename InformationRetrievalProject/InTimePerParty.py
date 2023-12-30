import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re

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
    'διότι', 'γιατί', 'προς', 'πω', 'κάθε', 'πιο', 'γι', '000',
    'εδώ', 'αυτές', 'εκεί', 'στους', 'οποίες', 'οποίοι', 'υπό', 'επί', 'όλα', 'όλοι',
    'εις', 'όσον', 'περί', 'πώς', 'ώστε', 'ήδη', 'διά', 'πα', 'σο', 'αρα', 'παρά', 'εγώ', 'μόνο',
    'όπου', 'τόσο', 'πάρα', 'δύο', 'πως', 'τότε', 'μα', 'κι', 'δισ', 'δις', 'αυτοί', 'εσείς', 'μέχρι', 'λόγω'

]

# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')

# Μετατροπή της στήλης ημερομηνίας σε datetime
df['sitting_date'] = pd.to_datetime(df['sitting_date'], format='%d/%m/%Y')

# Καθορισμός της δεκαετίας για κάθε ομιλία
def assign_decade(date):
    year = date.year
    if 1980 <= year <= 1990:
        return '1980-1990'
    elif 1991 <= year <= 2000:
        return '1991-2000'
    elif 2001 <= year <= 2010:
        return '2001-2010'
    elif 2011 <= year <= 2022:
        return '2011-2022'
    else:
        return 'Other'

df['decade'] = df['sitting_date'].apply(assign_decade)

# Συνάρτηση για τον καθαρισμό κειμένου
def clean_text(text):
    # Αφαιρούμε τα σημεία στίξης και τα αριθμητικά δεδομένα
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Διαχωρισμός των λέξεων και αφαίρεση των stop words
    words = text.split()
    words = [word for word in words if word not in greek_stop_words]

    # Επιστροφή του καθαρισμένου κειμένου
    return ' '.join(words)

df['cleaned_speech'] = df['speech'].apply(clean_text)
df = df[df['cleaned_speech'].str.strip() != '']

# Ομαδοποίηση των ομιλιών ανά βουλευτή και δεκαετία
grouped = df.groupby(['political_party', 'decade'])



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


# Εξαγωγή και εκτύπωση των κυριότερων λέξεων-κλειδιών για κάθε βουλευτή σε μια συγκεκριμένη δεκαετία
for (name, decade), group in grouped:
    keywords = extract_keywords(group['speech'])
    if keywords:
        print(f"Κυριότερες λέξεις-κλειδιά για το κόμμα {name} στη δεκαετία {decade}:")
        for keyword, score in keywords:
            print(f"{keyword}: {score}")
        print("\n")
    else:
        print(f"No keywords found for {name} in {decade}\n")
