
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
from feature_engineering.feature_url_email import extract_features

df = pd.read_csv("data/sample_data.csv")
X = extract_features(df)
y = df['label']
model = joblib.load("models/rf_model.pkl")

y_pred = model.predict(X)
y_score = model.predict_proba(X)[:, 1]

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("visualization/confusion_matrix.png")

fpr, tpr, _ = roc_curve(y, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig("visualization/roc_curve.png")
