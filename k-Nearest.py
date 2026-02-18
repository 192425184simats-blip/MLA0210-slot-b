import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("parking_dataset.csv")

# Encode and store encoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Split data
X = df.drop("Available", axis=1)
y = df["Available"]

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create new sample as DataFrame (IMPORTANT)
new_data = pd.DataFrame({
    "Time": ["Peak"],
    "DayType": ["Weekend"],
    "Occupancy": ["High"]
})

# Encode new sample using SAME encoders
for column in new_data.columns:
    new_data[column] = encoders[column].transform(new_data[column])

# Predict
prediction = knn.predict(new_data)

# Decode result
result = encoders["Available"].inverse_transform(prediction)

print("KNN Prediction:", result[0])
