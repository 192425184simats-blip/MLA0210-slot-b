
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = {
    'Income': [30000, 50000, 40000, 80000, 20000, 90000, 60000],
    'Age': [25, 35, 45, 50, 23, 55, 40],
    'Loan_Amount': [5000, 20000, 15000, 10000, 3000, 25000, 12000],
    'Late_Payments': [2, 0, 1, 0, 4, 0, 1],
    'Credit_Score': [0, 1, 0, 1, 0, 1, 1]  # 0 = Bad, 1 = Good
}

df = pd.DataFrame(data)
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_customer = np.array([[45000, 30, 12000, 1]])
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nCredit Score: GOOD")
else:
    print("\nCredit Score: BAD")
