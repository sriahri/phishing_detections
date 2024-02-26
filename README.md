# DETECTING PHISHING/SOCIAL ENGINEERING ATTACKS
## Phishing/Social engineering attack:

Phishing and social engineering attacks are types of cyber-attacks that are designed to trick people into revealing sensitive information or taking actions that they wouldn't normally take.
Phishing attacks typically involve an attacker sending a fraudulent email or message that appears to be from a legitimate source, such as a bank, social media platform, or online retailer. The message will usually contain a link or attachment that, when clicked, will take the victim to a fake website, or download malware onto their device. The fake website will often look very similar to the real website, and the victim will be prompted to enter their login credentials or other sensitive information.
Social engineering attacks, on the other hand, are more focused on manipulating people's emotions and behavior to gain access to sensitive information or systems. These attacks can take many forms, such as impersonating a trusted authority figure or using psychological tactics to convince the victim to divulge confidential information.
Both phishing and social engineering attacks can be very effective because they exploit human vulnerabilities and rely on people's trust and willingness to help. It's important to be vigilant and cautious when receiving unsolicited messages or requests for information, and to always verify the authenticity of the sender or website before providing any sensitive information.

## Forms of attack:
Phishing and social engineering attacks can take many different forms, but here are some of the most common types:
#### Email phishing: 
This is the most common form of phishing attack, where the attacker sends a fraudulent email that appears to be from a legitimate source, such as a bank or an online shopping site, to trick the recipient into providing sensitive information.
#### Spear phishing: 
This is a targeted phishing attack where the attacker focuses on a specific individual or organization, using information gathered from research or social media to create a more convincing email or message.
#### Smishing: 
This is a form of phishing that uses SMS text messages to trick the recipient into clicking on a link or providing sensitive information.
#### Vishing: 
This is a form of phishing that uses voice calls to trick the recipient into providing sensitive information or taking other actions.
#### Pretexting: 
This is a social engineering attack where the attacker creates a false scenario or pretext to gain the victim's trust and obtain sensitive information.
#### Baiting: 
This is a social engineering attack where the attacker leaves a tempting lure, such as a USB drive or email attachment, in a public place to entice the victim to pick it up and use it.
#### Watering hole attack: 
This is a targeted attack where the attacker compromises a legitimate website that the victim is likely to visit, to infect their device with malware or steal their login credentials.
## Phishing website detection:
Phishing website detection is the process of identifying and blocking websites that are designed to look like legitimate sites but are actually created by cybercriminals to trick users into providing sensitive information such as passwords, credit card numbers, or other personal data.
The code in the path models/Phishing_website_detection.py does the following.
1.	Imports necessary libraries such as pickle, pandas, RandomForestClassifier, accuracy_score, confusion_matrix, and train_test_split from sklearn.
2.	Reads a CSV file containing a dataset of phishing websites and stores it in a DataFrame called "df".
3.	Splits the dataset into training and testing sets using the train_test_split() function.
4.	Trains a random forest classifier on the training data using the RandomForestClassifier() function and the fit() method.
5.	Makes predictions on the test data using the predict() method and computes the accuracy of the classifier using the accuracy_score() function.
6.	Prints the accuracy score and confusion matrix of the classifier.
7.	Defines two functions to save and load the trained classifier model using the pickle library.
8.	The save_model() function saves the trained model to a file called "phishing_website_detector.pkl".
9.	The load_model() function loads the saved model from the file specified by the file_name argument.
This code determines that the website is either a legitimate website or not. It uses supervised learning.
Phishing Email detection:
	Phishing email detection is the process of identifying and blocking fraudulent emails that are designed to trick users into revealing sensitive information or taking other malicious actions. Phishing emails often appear to be from legitimate sources, such as banks, online retailers, or social media sites, and may contain urgent requests or threats to create a sense of urgency and pressure the recipient to take immediate action.
The code in the path models/phishing_email_detection.py does the following:
1.	Imports necessary libraries such as pandas, train_test_split, CountVectorizer, accuracy_score, confusion_matrix, RandomForestClassifier, Sequential, Dense, LSTM, GRU, and MultinomialNB.
2.	Reads a CSV file containing a dataset of emails and stores it in a DataFrame called "dataset".
3.	Removes duplicates from the dataset using the drop_duplicates() method.
4.	Checks if there are any null entries in the dataset using the isnull() method.
5.	Cleans the text data using natural language processing techniques such as stemming and removing stop words.
6.	Creates a bag-of-words model using the CountVectorizer() function and transforms the text data into numerical data.
7.	Splits the dataset into training and testing sets using the train_test_split() function.
8.	Trains a Multinomial Naive Bayes classifier on the training data using the fit() method.
9.	Makes predictions on the test data using the predict() method and computes the accuracy of the classifier using the accuracy_score() function and the confusion matrix using the confusion_matrix() function.
10.	Defines four functions to save and load the trained classifier model and the count vectorizer object using the pickle library.
11.	The save_model() function saves the trained model to a file called "phishing_email_detector.pkl".
12.	The save_count_vectorizer() function saves the count vectorizer object to a file called "phishing_email_detector_count_vectorizer.pkl".
13.	The load_model() function loads the saved model from the file specified by the file_name argument.
14.	The load_count_vectorizer() function loads the saved count vectorizer object from the file specified by the file_name argument.
Phishing SMS detection:
	Phishing SMS detection is the process of identifying and blocking fraudulent SMS (text) messages that are designed to trick users into revealing sensitive information or taking other malicious actions. Phishing SMS messages often appear to be from legitimate sources, such as banks, online retailers, or social media sites, and may contain urgent requests or threats to create a sense of urgency and pressure the recipient to take immediate action.
The code in the path models/spam_sms_detection.py does the following:
1.	Imports the necessary libraries such as pandas, CountVectorizer, accuracy_score, confusion_matrix, train_test_split, and MultinomialNB.
2.	Reads a CSV file containing SMS messages and stores it in a DataFrame called "sms_data".
3.	Removes unwanted columns from the DataFrame and renames the remaining columns.
4.	Maps the labels to binary values of 0 (ham) and 1 (spam).
5.	Converts the text to lowercase and tokenizes it using the CountVectorizer() function.
6.	Splits the dataset into training and testing sets using the train_test_split() function.
7.	Instantiates a Multinomial Naive Bayes classifier.
8.	Trains the classifier on the training data using the fit() method.
9.	Makes predictions on the test data using the predict() method.
10.	Calculates the accuracy score of the classifier using the accuracy_score() function and the confusion matrix using the confusion_matrix() function.
11.	Preprocesses the input data by converting it to lowercase, tokenizing it, and transforming it using the CountVectorizer() object.
12.	Uses the trained classifier to make a prediction on the preprocessed input data.
13.	Defines four functions to save and load the trained classifier model and the CountVectorizer() object using the pickle library.
14.	The save_model() function saves the trained model to a file called "spam_sms_detector.pkl".
15.	The load_model() function loads the saved model from the file specified by the file_name argument.
16.	The save_count_vectorizer() function saves the CountVectorizer() object to a file called "spam_sms_detector.pkl".
17.	The load_count_vectorizer() function loads the saved CountVectorizer() object from the file specified by the file_name argument.

## Integrating the models:
To integrate the models, we use the Flask framework of the python.
The code in the path app.py does the following. This code creates a Flask web application for detecting spam messages in both SMS and email formats. The code defines two functions for predicting the label of an input message: predict_label_EMAIL(email_content) for email messages and predict_label_SMS(sms_content) for SMS messages. Both functions take the input message content as an argument and use a trained model and a CountVectorizer object to predict whether the message is spam or not. The trained model and CountVectorizer object are loaded from saved files using the load_model() and load_count_vectorizer() functions defined in the code.
The code also defines four routes for the web application: the main page, an SMS check page, an email check page, and an about page. The main() function renders the main page, while the sms_check() and email_check() functions render the SMS and email check pages, respectively. The about_page() function renders an about page providing information about the spam detection model.
Finally, the get_output_email() and get_output_sms() functions handle the form submission from the respective pages and call the appropriate prediction function to predict whether the input message is spam or not. The predicted label is then returned to the respective page for display. The Flask app is run on the localhost on port 8080 with debug mode enabled.
## Datasets:
•	[Data mendely dataset](https://data.mendeley.com/datasets/72ptz43s9v/1)

•	[Phishing website detection dataset](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector?resource=download&select=phishing.csv)

•	[Yelp-Reviews dataset](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset)

•	[Fake Reviews Dataset](https://osf.io/3vds7)

•	[sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

•	[spam-or-ham-email-classification dataset](https://www.kaggle.com/code/balakishan77/spam-or-ham-email-classification/input)
