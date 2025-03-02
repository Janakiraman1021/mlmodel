import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load datasets
benign_df = pd.read_csv('1.Benign_list_big_final.csv')
phishing_df = pd.read_csv('4.phishing.csv')
valid_df = pd.read_csv('2.online-valid.csv')
legitimate_df = pd.read_csv('3.legitimate.csv')
urldata_df = pd.read_csv('5.urldata.csv')

# Combine datasets
benign_df['label'] = 0  # Not spam
phishing_df['label'] = 1  # Spam
combined_df = pd.concat([benign_df, phishing_df, valid_df, legitimate_df, urldata_df])

# Data cleaning and feature engineering
# (Assuming necessary preprocessing steps here)

# Split data
X = combined_df.drop('label', axis=1)
y = combined_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, 'spam_detection_model.pkl')