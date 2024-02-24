import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from InformationRetrievalProject.GreekStopWords import greek_stop_words  # Import your list of Greek stop words


# # Load the cleaned CSV file
df_cleaned = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Group the DataFrame by 'member_name'
grouped = df_cleaned.groupby('member_name')

# Define extract_keywords function
def extract_keywords(speeches):
    # Combine all speeches of a member into a single text
    combined_speeches = ' '.join(speeches)

    # Check if the combined speeches are empty after preprocessing
    if not combined_speeches.strip():
        return []

    try:
        vectorizer = CountVectorizer()
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
    except ValueError:
        print(f"Error occurred at group {name}")
        return []

#  Iterate over grouped DataFrame
for name, group in grouped:
    keywords = extract_keywords(group['cleaned_speech'])
    if keywords:
        print(f"Top keywords for group {name}:")
        for keyword, score in keywords:
            print(f"{keyword}: {score}")
        print("\n")
    else:
        print(f"No keywords found for group {name}\n")
