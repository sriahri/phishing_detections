# # Import libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, GRU
#
# # Load the data
# df = pd.read_csv('F:/UC_Denver/Emerging_System_Security/Final_Project/datasets/emails.csv')
# X = df['text']
# y = df['spam']
#
# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # Feature Engineering
# vect = CountVectorizer()
# X_train = vect.fit_transform(X_train)
# X_test = vect.transform(X_test)
#
# # Random Forest Model
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# rf_preds = rf_model.predict(X_test)
#
# # Neural Network Model
# lstm_model = Sequential()
# lstm_model.add(LSTM(128, input_dim=X_train.shape[1], recurrence_dropout=0.2, dropout=0.3, return_sequences=True))
# lstm_model.add(LSTM(64, return_sequences=False))
# lstm_model.add(Dense(1, activation='sigmoid'))
# lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
# lstm_model.fit(X_train, y_train, batch_size=32, epochs=10)
# lstm_preds = lstm_model.predict_classes(X_test)
#
# # Evaluating the models
# print('Random Forest Accuracy: ', accuracy_score(y_test, rf_preds))
# print('Random Forest Report: ', classification_report(y_test, rf_preds))
# print('LSTM Accuracy: ', accuracy_score(y_test, lstm_preds))
# print('LSTM Report: ', classification_report(y_test, lstm_preds))


import pickle
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('F:/UC_Denver/Emerging_System_Security/Final_Project/datasets/emails.csv')
print(dataset.columns)
print(dataset.shape)

# Checking for duplicates and removing them
dataset.drop_duplicates(inplace=True)
print(dataset.shape)

# Checking for any null entries in the dataset
print(pd.DataFrame(dataset.isnull().sum()))
'''
text  0
spam  0
'''
# Using Natural Language Processing to cleaning the text to make one corpus
# Cleaning the texts

# Every mail starts with 'Subject :' will remove this from each text
dataset['text'] = dataset['text'].map(lambda text: text[1:])
dataset['text'] = dataset['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ', text)).apply(
    lambda x: (x.lower()).split())
ps = PorterStemmer()
corpus = dataset['text'].apply(lambda text_list: ' '.join(list(map(lambda word: ps.stem(word), (
    list(filter(lambda text: text not in set(stopwords.words('english')), text_list)))))))

# Creating the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(corpus.values).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Fitting Naive Bayes classifier to the Training set
classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix


cm = confusion_matrix(y_test, y_pred)
'''
Confusion Matrix
array([[863,  11],
       [  1, 264]])
'''
# this function computes subset accuracy


accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred, normalize=False)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train,
                             cv=10)


def save_model():
    file_name = 'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/phishing_email_detector.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(classifier, file)


def save_count_vectorizer():
    file_name = 'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models' \
                '/phishing_email_detector_count_vectorizer.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(cv, file)


def load_model(file_name):
    with open(file_name, 'rb') as file:
        classifier = pickle.load(file)
    return classifier


def load_count_vectorizer(file_name):
    with open(file_name, 'rb') as file:
        count_vectorizer = pickle.load(file)
    return count_vectorizer


save_model()
