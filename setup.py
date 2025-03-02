import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import re
from urllib.parse import urlparse
import joblib

# Load datasets
benign_df = pd.read_csv('1.Benign_list_big_final.csv')
phishing_df = pd.read_csv('4.phishing.csv')

# Ensure 'url' column exists
if 'url' not in benign_df.columns:
    benign_df['url'] = ''  # Add a placeholder column if missing
if 'url' not in phishing_df.columns:
    phishing_df['url'] = ''  # Add a placeholder column if missing

# Labeling
benign_df['label'] = 0
phishing_df['label'] = 1

# Add new features to the dataset
def add_features(df):
    df['URL_Length'] = df['url'].apply(lambda x: len(x))
    df['URL_Depth'] = df['url'].apply(lambda x: len([i for i in x.split('/') if i]))
    df['Num_Dots'] = df['url'].apply(lambda x: x.count('.'))
    df['Has_IP'] = df['url'].apply(lambda x: 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', x) else 0)
    df['Has_At'] = df['url'].apply(lambda x: 1 if '@' in x else 0)
    df['Has_Hyphen'] = df['url'].apply(lambda x: 1 if '-' in x else 0)
    df['Num_Subdomains'] = df['url'].apply(lambda x: len(urlparse(x).netloc.split('.')))
    df['Is_Https'] = df['url'].apply(lambda x: 1 if urlparse(x).scheme == 'https' else 0)
    df['Has_Redirection'] = df['url'].apply(lambda x: 1 if '//' in x else 0)
    suspicious_words = ['login', 'secure', 'update', 'free', 'account', 'verify']
    df['Has_Suspicious_Word'] = df['url'].apply(lambda x: 1 if any(word in x for word in suspicious_words) else 0)
    df['Num_Slashes'] = df['url'].apply(lambda x: x.count('/'))
    df['Num_Question_Marks'] = df['url'].apply(lambda x: x.count('?'))
    df['Num_Equals'] = df['url'].apply(lambda x: x.count('='))
    df['Num_Percent'] = df['url'].apply(lambda x: x.count('%'))
    df['Num_Ampersand'] = df['url'].apply(lambda x: x.count('&'))
    
    # Add new features
    df['Domain_Length'] = df['url'].apply(lambda x: len(urlparse(x).netloc))
    df['Has_Suspicious_TLD'] = df['url'].apply(lambda x: 1 if any(urlparse(x).netloc.endswith(tld) for tld in ['.xyz', '.top', '.fit', '.rent', '.cheap']) else 0)
    df['Has_Numbers_In_Domain'] = df['url'].apply(lambda x: 1 if any(c.isdigit() for c in urlparse(x).netloc) else 0)
    
    # Expand suspicious words list
    suspicious_words.extend(['cheap', 'rooms', 'deal', 'best', 'discount', 'offer', 'buy'])
    df['Suspicious_Word_Count'] = df['url'].apply(lambda x: sum(1 for word in suspicious_words if word in x.lower()))
    
    return df

# Apply the feature extraction to both datasets
benign_df = add_features(benign_df)
phishing_df = add_features(phishing_df)

# Combine datasets
data = pd.concat([benign_df, phishing_df])

# Enhanced feature selection - add more relevant features if available
features = data[[
    'URL_Length', 
    'URL_Depth', 
    'Num_Dots',
    'Has_IP',
    'Has_At',
    'Has_Hyphen',
    'Num_Subdomains',
    'Is_Https',
    'Has_Redirection',
    'Has_Suspicious_Word',
    'Num_Slashes',
    'Num_Question_Marks',
    'Num_Equals',
    'Num_Percent',
    'Num_Ampersand',
    'Domain_Length',
    'Has_Suspicious_TLD',
    'Has_Numbers_In_Domain',
    'Suspicious_Word_Count'
]]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

labels = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, 
    labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)

# XGBoost parameters for maximum accuracy
xgb_params = {
    'max_depth': 12,  # Increased from 10
    'learning_rate': 0.01,
    'n_estimators': 4000,  # Increased from 3000
    'min_child_weight': 2,  # Increased from 1
    'gamma': 0.2,  # Increased from 0.1
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.2,  # Added to handle class imbalance
    'objective': 'binary:logistic',
    'random_state': 42,
    'eval_metric': 'logloss',
    'tree_method': 'hist'
}

# Model training with correct parameters
model = xgb.XGBClassifier(**xgb_params)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, 
    y_train,
    eval_set=eval_set,
    verbose=False,  # Change from True to False to stop printing training progress
    # # callbacks=[xgb.callback.EarlyStopping(
    #     rounds=50,
    #     # stopping_metric='logloss',  # Changed from 'metric' to 'stopping_metric'
    #     save_best=True
    # )]
)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nTest Set Metrics:")
print(classification_report(y_test, y_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Training accuracy
train_pred = model.predict(X_train)
print("\nTraining Set Metrics:")
print("Training Accuracy:", accuracy_score(y_train, train_pred))

def extract_features(url):
    """Extract features from a URL for prediction."""
    parsed_url = urlparse(url)
    
    # Calculate features similar to training data
    url_length = len(url)
    url_depth = len([x for x in parsed_url.path.split('/') if x]) if parsed_url.path else 0
    num_dots = url.count('.')
    has_ip = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0
    has_at = 1 if '@' in url else 0
    has_hyphen = 1 if '-' in url else 0
    num_subdomains = len(urlparse(url).netloc.split('.'))
    is_https = 1 if parsed_url.scheme == 'https' else 0
    has_redirection = 1 if '//' in parsed_url.path else 0
    suspicious_words = ['login', 'secure', 'update', 'free', 'account', 'verify']
    has_suspicious_word = 1 if any(word in url for word in suspicious_words) else 0
    num_slashes = url.count('/')
    num_question_marks = url.count('?')
    num_equals = url.count('=')
    num_percent = url.count('%')
    num_ampersand = url.count('&')
    
    # Add new feature calculations
    domain_length = len(parsed_url.netloc)
    has_suspicious_tld = 1 if any(parsed_url.netloc.endswith(tld) for tld in ['.xyz', '.top', '.fit', '.rent', '.cheap']) else 0
    has_numbers_in_domain = 1 if any(c.isdigit() for c in parsed_url.netloc) else 0
    suspicious_words.extend(['cheap', 'rooms', 'deal', 'best', 'discount', 'offer', 'buy'])
    suspicious_word_count = sum(1 for word in suspicious_words if word in url.lower())
    
    features_dict = {
        'URL_Length': [url_length],
        'URL_Depth': [url_depth],
        'Num_Dots': [num_dots],
        'Has_IP': [has_ip],
        'Has_At': [has_at],
        'Has_Hyphen': [has_hyphen],
        'Num_Subdomains': [num_subdomains],
        'Is_Https': [is_https],
        'Has_Redirection': [has_redirection],
        'Has_Suspicious_Word': [has_suspicious_word],
        'Num_Slashes': [num_slashes],
        'Num_Question_Marks': [num_question_marks],
        'Num_Equals': [num_equals],
        'Num_Percent': [num_percent],
        'Num_Ampersand': [num_ampersand],
        'Domain_Length': [domain_length],
        'Has_Suspicious_TLD': [has_suspicious_tld],
        'Has_Numbers_In_Domain': [has_numbers_in_domain],
        'Suspicious_Word_Count': [suspicious_word_count]
    }
    
    return pd.DataFrame(features_dict)

def predict_url(url, model, scaler):
    """Predict if a URL is spam or not."""
    # Extract features
    features = extract_features(url)
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    
    return {
        'url': url,
        'is_spam': bool(prediction),
        'confidence': float(probability[1] if prediction else probability[0])
    }

# Save the trained model and scaler for later use
joblib.dump(model, 'spam_detector_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')

# Example usage
if __name__ == "__main__":
    # Test some URLs
    test_urls = [
        "http://.com",
        "https://www.google.com",
        # Add more test URLs
    ]
    
    print("\nTesting URLs:")
    for url in test_urls:
        result = predict_url(url, model, scaler)
        print(f"\nURL: {result['url']}")
        print(f"Prediction: {'NOT SPAM' if result['is_spam'] else ' SPAM'}")
        print(f"Confidence: {result['confidence']:.2%}")