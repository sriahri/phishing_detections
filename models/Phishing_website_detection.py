import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('F:/UC_Denver/Emerging_System_Security/Final_Project/datasets/dataset_full_phishing_websites.csv')

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df.drop('phishing', axis=1),
                                                    df['phishing'], test_size=0.2)

# Train random forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# Save model
def save_model():
    file_name = 'F:/UC_Denver/Emerging_System_Security/Final_Project/saved_models/phishing_website_detector.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(rf, file)


def load_model(file_name):
    with open(file_name, 'rb') as file:
        classifier = pickle.load(file)
    return classifier


save_model()
