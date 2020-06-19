import pandas as pd
import numpy as np
import nltk
import sklearn
df = pd.read_csv('/home/ankitsingh/Desktop/datasets/spam.csv', encoding='latin-1', sep=',')
df.dropna(axis=1, inplace=True)
label = df.v1
messages = df.v2
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label = le.fit_transform(label)

# use regular expressions to replace email addresses, URLs, phone numbers, other numbers

# Replace email addresses with 'email'
processed = messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

processed = processed.str.lower()

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
processed = processed.apply(lambda rec: ' '.join(word for word in rec.split(' ') if word not in stopwords))

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
processed = processed.apply(lambda rec: ' '.join(ps.stem(word) for word in rec.split()))

from nltk.tokenize import word_tokenize
all_words = []
for message in processed:
    words = word_tokenize(message)
    for word in words:
        all_words.append(word)
all_words = nltk.FreqDist(all_words)

def find_features(message):
    features = {}
    words = word_tokenize(message)
    for word in all_words:
        features[word] = True if word in words else False
    return features

messages = zip(processed, label)
feature_set = [(find_features(text), label) for (text, label) in messages]

from sklearn.model_selection import train_test_split
(training, testing) = train_test_split(feature_set, test_size = 0.25)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC())
model.train(training)
accuracy = nltk.classify.accuracy(model, testing)

print('Accuracy with SVM model: {}'.format(accuracy))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

names = ['KNeighborsClassifier', 'LogisticRegression', 'SGDClassifier', 'DecisionTreeClassifier', 
         'RandomForestClassifier']
models = [KNeighborsClassifier(), LogisticRegression(), SGDClassifier(), DecisionTreeClassifier(), 
         RandomForestClassifier()]
all_models = zip(names, models)

for name, model in all_models:
    classifier = SklearnClassifier(model)
    classifier.train(training)
    print('Accuracy with {} model is {}'
          .format(name, nltk.classify.accuracy(classifier, testing)*100))
          
from sklearn.ensemble import VotingClassifier
names = ['KNeighborsClassifier', 'LogisticRegression', 'SGDClassifier', 'DecisionTreeClassifier', 
         'RandomForestClassifier']
models = [KNeighborsClassifier(), LogisticRegression(), SGDClassifier(), DecisionTreeClassifier(), 
         RandomForestClassifier()]
all_models = tuple(zip(names, models))
model = SklearnClassifier(VotingClassifier(estimators=all_models, voting='hard', n_jobs=-1))
model.train(training)
print('Accuracy with voting classifier {}'.format(nltk.classify.accuracy(model, testing)))
