import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("weather_nb.csv")

le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

model = GaussianNB()
model.fit(X, y)

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)
