import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sms_data = pd.read_csv('F:/UC_Denver/Emerging_System_Security/Final_Project/datasets/SMSSpamCollection.csv', sep='::',
                       engine='python')

# print(sms_data)

# Remove unwanted columns
# sms_data = sms_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Rename columns
sms_data.columns = ['label', 'text']

# Map labels to binary values
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Convert text to lowercase
sms_data['text'] = sms_data['text'].str.lower()

# Tokenize text
vectorizer = CountVectorizer(stop_words='english')
sms_data['text'] = vectorizer.fit_transform(sms_data['text']).toarray()

print(sms_data.shape)

X_train, X_test, y_train, y_test = train_test_split(sms_data['text'], sms_data['label'], test_size=0.2, random_state=42)

X_train = X_train.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy().reshape(-1, 1)
# Instantiate a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier on the training data

clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_matrix)

# Preprocess the input data
input_text = "Hello Everyone."
input_text = input_text.lower()
input_text = vectorizer.transform([input_text]).toarray()
input_text = input_text.reshape(-1, 1)

# Use the trained model to make a prediction
prediction = clf.predict(input_text)
print("Prediction:", prediction[0])


def save_model():
    file_name = 'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/spam_sms_detector.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(clf, file)


def load_model(file_name):
    with open(file_name, 'wb') as file:
        classifier = pickle.load(file)
    return classifier


def save_count_vectorizer():
    file_name = 'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/spam_sms_detector.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(vectorizer, file)


def load_count_vectorizer(file_name):
    with open(file_name, 'rb') as file:
        count_vectorizer = pickle.load(file)
    return count_vectorizer


save_model()
