import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Age": [25, 30, 35, 40, 28, 32, 45, 50],
    "Income": [40000, 50000, 60000, 80000, 45000, 52000, 90000, 100000],
    "CreditScore": [650, 700, 750, 800, 680, 720, 820, 850],
    "AdBudget": [10000, 15000, 20000, 25000, 12000, 18000, 30000, 35000],
    "PreviousSales": [200, 300, 400, 500, 250, 350, 600, 700],
    "Region": ["Rural", "Urban", "Semi-Urban", "Urban",
               "Rural", "Semi-Urban", "Urban", "Urban"],
    "FutureSales": [220, 320, 420, 520, 270, 370, 630, 750]
}

df = pd.DataFrame(data)

X = df.drop("FutureSales", axis=1)
y = df["FutureSales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Region"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame({
    "Age": [30],
    "Income": [55000],
    "CreditScore": [720],
    "AdBudget": [16000],
    "PreviousSales": [350],
    "Region": ["Urban"]
})

prediction = model.predict(sample)

print("Predicted Future Sales:", prediction[0])
