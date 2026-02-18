import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("parking_dataset.csv")

# Encode and store encoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Split features and target
X = df.drop("Available", axis=1)
y = df["Available"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Create new sample as DataFrame (IMPORTANT)
new_data = pd.DataFrame({
    "Time": ["Peak"],
    "DayType": ["Weekend"],
    "Occupancy": ["High"]
})

# Encode using SAME encoders
for column in new_data.columns:
    new_data[column] = encoders[column].transform(new_data[column])

# Predict
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

# Decode output
result = encoders["Available"].inverse_transform(prediction)

print("Logistic Prediction:", result[0])
print("Availability Probability:", probability)
