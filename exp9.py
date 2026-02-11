import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("position_salary.csv")

X = data[['Level']]
y = data['Salary']

linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

print("Linear Regression R2 Score:")
print(r2_score(y, linear_pred))

print("\nPolynomial Regression R2 Score:")
print(r2_score(y, poly_pred))
