import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load datasets
benign_df = pd.read_csv("1.Benign_list_big_final.csv", header=None, names=["URL"])
phishing_df1 = pd.read_csv("2.online-valid.csv")
phishing_df2 = pd.read_csv("4.phishing.csv")
legitimate_df = pd.read_csv("3.legitimate.csv")
urldata_df = pd.read_csv("5.urldata.csv")

# Label phishing datasets
phishing_df1["Label"] = 1
phishing_df2["Label"] = 1
urldata_df["Label"] = urldata_df["Label"].apply(lambda x: 1 if x == "phishing" else 0)
legitimate_df["Label"] = 0

# Selecting common columns
common_cols = ["Domain", "Have_IP", "Have_At", "URL_Length", "URL_Depth", "Redirection",
               "https_Domain", "TinyURL", "Prefix/Suffix", "DNS_Record", "Web_Traffic",
               "Domain_Age", "Domain_End", "iFrame", "Mouse_Over", "Right_Click", "Web_Forwards", "Label"]

# Concatenating data
df = pd.concat([phishing_df2[common_cols], legitimate_df[common_cols], urldata_df[common_cols]])

# Handle missing values
df.fillna(0, inplace=True)

# Encode categorical features if needed
encoder = LabelEncoder()
df["Domain"] = encoder.fit_transform(df["Domain"])

# Split data
X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
