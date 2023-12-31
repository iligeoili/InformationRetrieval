import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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



# Φόρτωση των δεδομένων
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020.csv')

# Εξαγωγή διανυσμάτων χαρακτηριστικών για κάθε ομιλία με TF-IDF
tfidf = TfidfVectorizer(stop_words=greek_stop_words)
tfidf_matrix = tfidf.fit_transform(df['speech'])



# Εφαρμογή LSI
lsi = TruncatedSVD(n_components=100)  # Ο αριθμός των διαστάσεων στον χώρο χαμηλότερων διαστάσεων
lsi_matrix = lsi.fit_transform(tfidf_matrix)

# Απόκτηση των σημαντικότερων θεματικών περιοχών
terms = tfidf.get_feature_names_out()
for i, component in enumerate(lsi.components_):
    terms_component = zip(terms, component)
    sorted_terms = sorted(terms_component, key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:10]
    print(f"Topic {i}:")
    for term in top_terms:
        print(term)
    print("\n")

# Προαιρετικά: Απεικόνιση των ομιλιών σε 2D χώρο
plt.scatter(lsi_matrix[:, 0], lsi_matrix[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('LSI 2D Representation of Speeches')
plt.show()