import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("mobile_price_dataset.csv")

X = df.drop("Price_Range", axis=1)
y = df["Price_Range"]

le_connect = LabelEncoder()
le_price = LabelEncoder()

X["Connectivity"] = le_connect.fit_transform(X["Connectivity"])
y = le_price.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

sample = [[2200, 6, 128, 6.4, 32, le_connect.transform(["5G"])[0]]]
prediction = model.predict(sample)

print("Predicted Price Range:", le_price.inverse_transform(prediction)[0])
