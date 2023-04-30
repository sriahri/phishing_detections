import pickle
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# load the saved model and count vectorizer
model = pickle.load(
    open('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/phishing_email_detector.pkl', 'rb'))
cv = pickle.load(open(
    'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/phishing_email_detector_count_vectorizer.pkl',
    'rb'))

# pre-process the text message
text = "Hello, this is a new email that I received. Please click on this link to claim your prize!"
text = re.sub('[^a-zA-Z0-9]+', ' ', text)
text = text.lower().split()
ps = PorterStemmer()
text = ' '.join(list(map(lambda word: ps.stem(word), (
    list(filter(lambda text: text not in set(stopwords.words('english')), text))))))

# transform the pre-processed text message using the count vectorizer
text_transformed = cv.transform([text]).toarray()

# use the saved model to predict the label of the text message
prediction = model.predict(text_transformed)
print(prediction)

if prediction[0] == 1:
    print('Spam')
else:
    print('Not Spam')
