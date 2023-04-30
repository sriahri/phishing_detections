import re
import string

import nltk

from models.spam_sms_detection import load_model, load_count_vectorizer

stopword = nltk.corpus.stopwords.words('english')


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)

    text = [word for word in tokens if word not in stopword]
    return text


classifier = load_model('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/spam_sms_detector.pkl')
count_vectorizer = load_count_vectorizer('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models'
                                         '/spam_sms_detector_count_vectorizer.pkl')

message = "Congratulations! You have been selected to receive a free gift. Please click on the link to claim your " \
          "prize."

message = re.sub('[^a-zA-Z0-9]+', ' ', message)
message = clean_text(message)

vector = count_vectorizer.transform(message)

prediction = classifier.predict(vector.toarray())

if prediction[0] == 1:
    print("Spam message")
else:
    print("Legitimate message")
