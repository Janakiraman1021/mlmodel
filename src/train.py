import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load datasets
urldata = pd.read_csv('5.urldata.csv')
benign = pd.read_csv('1.Benign_list_big_final.csv')
phishing = pd.read_csv('4.phishing.csv')

# Data preparation
urldata['Label'] = urldata['Label'].map({0: 'benign', 1: 'spam'})  # Assuming 0 is benign and 1 is spam
benign['Label'] = 'benign'
phishing['Label'] = 'spam'

# Combine datasets
combined_data = pd.concat([benign, phishing, urldata], ignore_index=True)

# Feature selection
features = combined_data[['URL_Length', 'URL_Depth', 'Redirection', 'https_Domain', 'TinyURL', 'DNS_Record', 'Web_Traffic', 'Domain_Age', 'iFrame', 'Mouse_Over', 'Right_Click', 'Web_Forwards']]
labels = combined_data['Label']

# Encoding categorical features
encoder = OneHotEncoder()
features_encoded = encoder.fit_transform(features).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))