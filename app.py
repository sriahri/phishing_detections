# from keras.models import load_model
import re
import string

import nltk
import numpy as np
from flask import Flask, render_template, request

from models.phishing_email_detection import load_model, load_count_vectorizer

# from PIL import Image

app = Flask(__name__)

dic = ['SPAM', 'NOT A SPAM']
x = np.array(dic)


def predict_label_EMAIL(email_content):
    # load the saved model and count vectorizer
    model = load_model('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/phishing_email_detector.pkl')
    cv = load_count_vectorizer('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models'
                               '/phishing_email_detector_count_vectorizer.pkl')

    # model.make_predict_function()

    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # pre-process the text message
    text = email_content
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
        return 'Spam'
    else:
        return 'Not Spam'


stopword = nltk.corpus.stopwords.words('english')


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)

    text = [word for word in tokens if word not in stopword]
    return text


def predict_label_SMS(sms_content):
    # load the saved model and count vectorizer
    model = load_model('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/spam_sms_detector.pkl')
    count_vectorizer = load_count_vectorizer('F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models'
                                             '/spam_sms_detector_count_vectorizer.pkl')

    # model.make_predict_function()

    import re

    # pre-process the text message
    message = sms_content

    message = re.sub('[^a-zA-Z0-9]+', ' ', message)
    message = clean_text(message)

    vector = count_vectorizer.transform(message)

    prediction = model.predict(vector.toarray())

    if prediction[0] == 1:
        return "Spam message"
    else:
        return "Legitimate message"


# routes

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/sms", methods=['GET', 'POST'])
def sms_check():
    return render_template("index_sms.html")


@app.route("/email", methods=['GET', 'POST'])
def email_check():
    return render_template("index_email.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output_email():
    if request.method == 'POST':
        email_content = request.form['email_content']
        p = predict_label_EMAIL(email_content)

    return render_template("index_email.html", prediction=p, email_content=email_content)


@app.route("/enter", methods=['GET', 'POST'])
def get_output_sms():
    if request.method == 'POST':
        sms_content = request.form['sms_content']
        p = predict_label_SMS(sms_content=sms_content)

    return render_template("index_sms.html", prediction=p, sms_content=sms_content)


if __name__ == '__main__':
    # app.debug = True
    app.run(port=8080, debug=False, host='0.0.0.0')
# app.run(debug = True)
