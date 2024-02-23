import pandas as pd
import spacy
from InformationRetrievalProject.GreekStopWords import greek_stop_words  # Import your list of Greek stop words
# Load the Greek language model
e=1
nlp = spacy.load("el_core_news_sm")
e += 1
print(e)




def preprocess_text(text):
    # Tokenize and process text with spaCy
    global e
    doc = nlp(text)
    e += 1
    print(e)
    # Remove verbs, numbers, and stop words
    filtered_tokens = [token.text for token in doc if token.pos_ != 'VERB' and not token.like_num and token.text not in greek_stop_words]
    return " ".join(filtered_tokens)
e += 1
print(e)
# Load the CSV file
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')
e += 1
print(e)
# Remove rows with NaN values in the 'speech' column
df_cleaned = df.dropna(subset=['speech'])
e += 1
print(e)
# Preprocess the 'speech' column
df_cleaned['cleaned_speech'] = df_cleaned['speech'].apply(preprocess_text)
e += 1
print(e)
# Remove rows where 'cleaned_speech' is empty after preprocessing
df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']

# Drop the column containing the old speeches
df_cleaned.drop(columns=['speech'], inplace=True)

# Remove rows where 'cleaned_speech' is empty after preprocessing
df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']
# Remove rows with special characters only in the 'cleaned_speech' column
df_cleaned = df_cleaned[~df_cleaned['cleaned_speech'].str.match(r'^[.,\s]*$')]

print(e)
# Save the cleaned DataFrame back to a new CSV file
df_cleaned.to_csv('Greek_Parliament_Proceedings_1989_2020_cleaned.csv', index=False)
