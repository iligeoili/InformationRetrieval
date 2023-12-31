from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import  TruncatedSVD

# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')

# Ο πίνακας με τα stop words
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
    'όπου', 'τόσο', 'πάρα', 'δύο', 'πως', 'τότε', 'μα', 'κι', 'δισ', 'δις', 'αυτοί', 'εσείς', 'μέχρι', 'λόγω',
    'πρέπει', 'μπορούμε', 'μπορεί', 'έχουμε', 'έχει', 'υπάρχει',
    'πρόκειται', 'λόγος', 'περίπτωση', 'μέρος', 'πρόβλημα',
    'επίσης', 'όπως', 'διότι', 'οπότε', 'επομένως', 'παρόλα',
    'όμως', 'παρότι', 'διαφορετικά', 'ακόμα', 'αυτή', 'αυτό',
    'αυτούς', 'αυτών', 'εκείνοι', 'εκείνων', 'εκείνη', 'εκείνο',
    'τέτοια', 'τέτοιο', 'τέτοιος', 'τέτοιων', 'σήμερα', 'χθες',
    'αύριο', 'προχθές', 'εδώ', 'εκεί', 'πουθενά', 'παντού', 'αλλού',
    'πάντα', 'ποτέ', 'συχνά', 'σπάνια', 'πρόσφατα', 'παλιά',
    'νωρίς', 'αργά', 'γρήγορα', 'αργά', 'καλά', 'κακά', 'καλύτερα',
    'χειρότερα', 'μεγάλο', 'μικρό', 'μεγαλύτερο', 'μικρότερο' ,
    'πλειοψηφία', 'δεκτό', 'άρθρο', 'ερωτάται', 'συνεπώς', 'έγινε',
    'γίνεται', 'σώμα', 'τροποποιήθηκε', 'κύριο', 'ναι', 'λόγο', 'πρόεδρε',
    'παρακαλώ', 'υπουργέ', 'όχι', 'ορίστε', 'κυρία', 'υπουργός', 'κύριος',
    'ευχαριστούμε', 'μάλιστα', 'πρέπει', 'υπάρχει', 'σήμερα', 'μπορεί',
    'συνάδελφε', 'διακόπτετε', 'συνάδελφοι', 'τελειώνετε', 'συνεχίστε',
    'ησυχία', 'παρών', 'υφυπουργός', 'ήθελα', 'μπορώ', 'επιτρέπετε', 'λεπτό',
    'προσωπικού', 'θέλω', 'ευχαριστώ', 'τρία', 'πέντε', 'δέκα', 'οκτώ', 'επτά', 'έξι' ,
    'λεπτά', 'τώρα', 'θέμα', 'κυβέρνηση', 'νομοσχέδιο', 'υπουργό',
    'συζήτηση', 'αφορά', 'γίνει', 'πολιτική', 'ολοκληρώστε', 'υφυπουργέ',
    'λαφαζάνη', 'δευτερολογία', 'χρόνο', 'σκέψη', 'καλοσύνη', 'χρόνος',
    'γεωργιάδη', 'κοινοβουλευτικός', 'εκπρόσωπος', 'δευτερολογήσει',
    'βουλευτής', 'δεκαπέντε', 'δώσω', 'τμήμα', 'δεκτή', 'αριθμό',
    'ερώτηση', 'επίκαιρη', '10', '11', 'γενικό', 'ειδικό',
    'εντάσσεται', 'ίδιο', 'τροπολογίας', 'τροπολογίες'

]

# Φτιάξτε τον TfidfVectorizer με τον πίνακα των stop words
tfidf = TfidfVectorizer(stop_words=greek_stop_words)

# Εφαρμόστε τον TF-IDF Vectorizer στα δεδομένα
tfidf_matrix = tfidf.fit_transform(df['speech'])
# Scaling the TF-IDF matrix (optional, but can improve results)
X_scaled = StandardScaler(with_mean=False).fit_transform(tfidf_matrix)

# Applying DBSCAN
# Note: You might need to adjust these parameters based on your data
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Add cluster information to your DataFrame
df['cluster'] = clusters

# Visualization with TruncatedSVD
svd = TruncatedSVD(n_components=2)
tfidf_matrix_2d = svd.fit_transform(X_scaled)

# Create the scatter plot
plt.scatter(tfidf_matrix_2d[:, 0], tfidf_matrix_2d[:, 1], c=clusters, cmap='rainbow')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('TF-IDF 2D Representation of Speeches with DBSCAN Clusters')
plt.show()
