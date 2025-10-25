import pandas as pd
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
def evaluate_model(name, y_true, y_pred):
    print(f"\n==== {name} ====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

csv_path = BASE_DIR / "UCI_Credit_Card.csv"
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)
df = df.drop(columns='ID')
df = df.rename(columns={'default.payment.next.month': 'default'})
x = df.drop(columns="default")
y = df["default"]
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
lr = LogisticRegression(max_iter=2000,class_weight='balanced')
lr.fit(X_train_scaled, y_train)
dt=DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train_scaled,y_train)
rf=RandomForestClassifier(class_weight='balanced')
rf.fit(X_train_scaled,y_train)
pred_lr = lr.predict(X_test_scaled)
pred_dt = dt.predict(X_test_scaled)
pred_rf = rf.predict(X_test_scaled)

evaluate_model("Logistic Regression", y_test, pred_lr)
evaluate_model("Decision Tree", y_test, pred_dt)
evaluate_model("Random Forest", y_test, pred_rf)

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(dt, X_test_scaled, y_test, ax=ax, name="Decision Tree")
RocCurveDisplay.from_estimator(rf, X_test_scaled, y_test, ax=ax, name="Random Forest")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
roc_plot_path = BASE_DIR / 'roc_curves.png'
plt.savefig(roc_plot_path)
print(f"\nROC curves saved to: {roc_plot_path}")
plt.close() 
print("LogReg AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]))
print("Decision Tree AUC:", roc_auc_score(y_test, dt.predict_proba(X_test_scaled)[:,1]))
print("Random Forest AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]))
model_path = BASE_DIR / 'credit_default_model.pkl'
scaler_path = BASE_DIR / 'scaler_model.pkl'
print(f"\nSaving model to: {model_path}")
print(f"Saving scaler to: {scaler_path}")
joblib.dump(rf, model_path)
joblib.dump(scaler, scaler_path)
print("\n Models saved successfully")
