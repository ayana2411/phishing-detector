
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from feature_engineering.feature_url_email import extract_features

df = pd.read_csv("data/sample_data.csv")
X = extract_features(df)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
lr = LogisticRegression()

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

import os
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(lr, "models/lr_model.pkl")

print("Random Forest Report:")
print(classification_report(y_test, rf.predict(X_test)))

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr.predict(X_test)))
