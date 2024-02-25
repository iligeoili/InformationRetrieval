import pandas as pd
import spacy
from InformationRetrievalProject.GreekStopWords import greek_stop_words
import re
flag = 0
# Load the Greek language model
nlp = spacy.load("el_core_news_sm")

# Load the cleaned DataFrame
df_cleaned = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Function to remove verbs from the cleaned text
def remove_verbs(text):
    doc = nlp(text)
    global flag
    flag +=1
    print("in remove verbs "+ str(flag))
    # Remove verbs from the text
    filtered_tokens = [token.text for token in doc if token.pos_ != 'VERB']
    return " ".join(filtered_tokens)

# Apply verb removal
df_cleaned['cleaned_speech'] = df_cleaned['cleaned_speech'].apply(remove_verbs)

# Function to refine the cleaned text further
def refine_cleaned_text(text):
    global flag
    flag += 1
    print("in refined_cleaned_text " + str(flag))
    # Normalize case for a case-insensitive match and split the text into tokens
    tokens = text.lower().split()
    # Remove any newly added stop words, considering case sensitivity
    refined_tokens = [token for token in tokens if token not in greek_stop_words]
    # Join tokens back and remove specific symbols if necessary
    cleaned_text = " ".join(refined_tokens)
    # Optionally, remove punctuation here if needed using regex substitution
    return re.sub(r'[.,;`-]+', '', cleaned_text)  # Adjust regex as needed to target specific symbols

# Apply the updated preprocessing
df_cleaned['cleaned_speech'] = df_cleaned['cleaned_speech'].apply(refine_cleaned_text)

# Further filter and clean
df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']
df_cleaned = df_cleaned[~df_cleaned['cleaned_speech'].str.match(r'^[.,\s]*$')]

# Save the DataFrame
df_cleaned.to_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv', index=False)
