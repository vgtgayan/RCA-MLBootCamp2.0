
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
import re  
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


import nltk
import ssl
 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stops = set(stopwords.words("english"))
print(stops)

import pandas as pd

model_name = "test-data-54000"

df = pd.read_csv(model_name+'.csv')
df.head(10)
# Create corpus from data frame
corpus = df.to_dict('records')


def clean_text(text, remove_stopwords=True):
    # Clean the text, with the option to remove stopwords.
    
    # Convert words to lower case and split them
    words = text.lower().split()

    # Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    cleaned_text = " ".join(words)

    # Clean the text
    cleaned_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", cleaned_text)
    cleaned_text = re.sub(r"\'s", " 's ", cleaned_text)
    cleaned_text = re.sub(r"\'ve", " 've ", cleaned_text)
    cleaned_text = re.sub(r"n\'t", " 't ", cleaned_text)
    cleaned_text = re.sub(r"\'re", " 're ", cleaned_text)
    cleaned_text = re.sub(r"\'d", " 'd ", cleaned_text)
    cleaned_text = re.sub(r"\'ll", " 'll ", cleaned_text)
    cleaned_text = re.sub(r",", " ", cleaned_text)
    cleaned_text = re.sub(r"\.", " ", cleaned_text)
    cleaned_text = re.sub(r"!", " ", cleaned_text)
    cleaned_text = re.sub(r"\(", " ( ", cleaned_text)
    cleaned_text = re.sub(r"\)", " ) ", cleaned_text)
    cleaned_text = re.sub(r"\?", " ", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    
    words = re.split('\W+', cleaned_text)                        # remove all non-words (make a list)
#     words = cleaned_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    cleaned_text = " ".join(stemmed_words)
    
    # Return cleaned text 
    return(cleaned_text)


# Input: corpus is a list of dictionaries(with all fields and data) 
def extract_clean_documents_from_corpus(corpus):
    print("Extracting and Cleaning documents...")
    final_corpus = []
    list_of_docs = []
    i = 0
    for ticket_dict in corpus:
        print("Processing ",ticket_dict['summary'])
        doc_cleaned_text = ''
        document_of_words = (str(ticket_dict['summary'])+" "+str(ticket_dict['description']))
#         document_of_words = (str(ticket_dict['question1']))
        doc_cleaned_text = clean_text(document_of_words)
        list_of_docs.append(doc_cleaned_text)
        final_corpus.append({'key':ticket_dict['key'],'index':i})
#         final_corpus.append({'qid1':ticket_dict['qid1'], 'words':doc_cleaned_text, 'index':i})
        i+=1
    return list_of_docs,final_corpus


# extract_clean_documents_from_corpus
list_of_docs,training_ticket_corpus = extract_clean_documents_from_corpus(corpus)


#save and load moddel
def save_model(model,model_name):
    pickle.dump(model, open(model_name, "wb"))

def load_model(model_file_path):
    with open(model_file_path, 'rb') as pickled_file:
        loaded_model_data = pickle.load(pickled_file)
    return loaded_model_data


def find_top_n_similar_documents(n,tfidf_test,tfidf_trainingset,cleaned_training_corpus):
   cosine_similarities = linear_kernel(tfidf_test, tfidf_trainingset).flatten()
   related_docs_indices = cosine_similarities.argsort()[:-n:-1]
   related_jira_ids = []
   for ticket in cleaned_training_corpus:
       if(ticket['index'] in related_docs_indices):
           related_jira_ids.append(ticket['key'])
            #  related_jira_ids.append(ticket['qid1'])
   return related_docs_indices,related_jira_ids



# Train the model
tfidf_model = TfidfVectorizer()

tfidf_trainingset = tfidf_model.fit_transform(list_of_docs)

save_model(tfidf_trainingset,model_name+".pickle")
save_model(training_ticket_corpus,model_name+"_corpus.pickle")



my_model = load_model(model_name+".pickle")
my_corpus = load_model(model_name+"_corpus.pickle")

tickets_dev_set = corpus[0:5]

for ticket in tickets_dev_set:
    
    document_test = (str(ticket['summary'])+" "+str(ticket['description']))
    cleaned_document = clean_text(document_test)
    cleaned_document = [cleaned_document]
    tfidf_test = tfidf_model.transform(cleaned_document)
    related_indices, related_jiras = find_top_n_similar_documents(6,tfidf_test[0:1],my_model,my_corpus)
    print("\n",ticket['key'],">>>> \n",related_jiras,"\n")