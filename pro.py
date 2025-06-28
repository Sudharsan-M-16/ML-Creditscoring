import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score,RocCurveDisplay
df = pd.read_csv("UCI_Credit_Card.csv")
df = df.drop(columns='ID')
df = df.rename(columns={'default.payment.next.month': 'default'})
x = df.drop(columns="default")
y = df["default"]
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pred_lr = lr.predict(X_test_scaled)
pred_dt = dt.predict(x_test)
pred_rf = rf.predict(x_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))
print("Classification Report:\n", classification_report(y_test, pred_lr))
RocCurveDisplay.from_estimator(lr, x_test, pred_lr)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)