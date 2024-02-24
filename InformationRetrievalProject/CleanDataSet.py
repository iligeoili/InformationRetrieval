import pandas as pd
import spacy
from InformationRetrievalProject.GreekStopWords import greek_stop_words  # Import your list of Greek stop words
# Load the Greek language model
e=1
nlp = spacy.load("el_core_news_sm")
e += 1
print(e)




# def preprocess_text(text):
#     # Tokenize and process text with spaCy
#     global e
#     doc = nlp(text)
#     e += 1
#     print(e)
#     # Remove verbs, numbers, and stop words
#     filtered_tokens = [token.text for token in doc if token.pos_ != 'VERB' and not token.like_num and token.text not in greek_stop_words]
#     return " ".join(filtered_tokens)
# e += 1
# print(e)
# # Load the CSV file
# df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')
# e += 1
# print(e)
# # Remove rows with NaN values in the 'speech' column
# df_cleaned = df.dropna(subset=['speech'])
# e += 1
# print(e)
# # Preprocess the 'speech' column
# df_cleaned['cleaned_speech'] = df_cleaned['speech'].apply(preprocess_text)
# e += 1
# print(e)
# # Remove rows where 'cleaned_speech' is empty after preprocessing
# df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']
#
# # Drop the column containing the old speeches
# df_cleaned.drop(columns=['speech'], inplace=True)
#
# # Remove rows where 'cleaned_speech' is empty after preprocessing
# df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']
# # Remove rows with special characters only in the 'cleaned_speech' column
# df_cleaned = df_cleaned[~df_cleaned['cleaned_speech'].str.match(r'^[.,\s]*$')]
#
# print(e)
# # Save the cleaned DataFrame back to a new CSV file
# df_cleaned.to_csv('Greek_Parliament_Proceedings_1989_2020_cleaned.csv', index=False)









df_cleaned = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')
e=0
# def refine_cleaned_text(text):
#     # Split the cleaned text back into tokens
#     tokens = text.split()
#     global e
#     e += 1
#     print(e)
#     # Remove any newly added stop words
#     refined_tokens = [token for token in tokens if token not in greek_stop_words]
#     return " ".join(refined_tokens)
#
# # Assuming df_cleaned is your DataFrame with the cleaned_speech column
# df_cleaned['cleaned_speech'] = df_cleaned['cleaned_speech'].apply(refine_cleaned_text)
#
# # Remove rows where 'cleaned_speech' is empty after further cleaning
# df_cleaned = df_cleaned[df_cleaned['cleaned_speech'].str.strip() != '']
# df_cleaned = df_cleaned[~df_cleaned['cleaned_speech'].str.match(r'^[.,\s]*$')]
#
# # Save the further cleaned DataFrame back to a new CSV file, if desired
# df_cleaned.to_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv', index=False)



########

import re
# Assuming greek_stop_words is correctly imported and updated

# Update the refine_cleaned_text function to handle case sensitivity and remove punctuation
def refine_cleaned_text(text):
    global e
    e += 1
    print(e)
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


