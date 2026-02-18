import pandas as pd
from sklearn.mixture import GaussianMixture

data = pd.read_csv("em_data.csv")

X = data[['X', 'Y']].values

model = GaussianMixture(n_components=3, random_state=42)
model.fit(X)

labels = model.predict(X)

data['Cluster'] = labels

print("Clustered Data:")
print(data)
