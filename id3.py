import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("parking_dataset.csv")

# Store encoders separately
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Split data
X = df.drop("Available", axis=1)
y = df["Available"]

# Train model
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Create new sample properly
new_data = pd.DataFrame({
    "Time": ["Peak"],
    "DayType": ["Weekend"],
    "Occupancy": ["High"]
})

# Encode new sample using SAME encoders
for column in new_data.columns:
    new_data[column] = encoders[column].transform(new_data[column])

# Predict
prediction = model.predict(new_data)

# Decode output
result = encoders["Available"].inverse_transform(prediction)

print("ID3 Prediction:", result[0])
