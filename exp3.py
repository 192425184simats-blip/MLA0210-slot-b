import pandas as pd
import numpy as np

data = pd.read_csv("play_tennis.csv")

def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    ent = 0
    for i in range(len(values)):
        p = counts[i] / np.sum(counts)
        ent -= p * np.log2(p)
    return ent

def information_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values, counts = np.unique(data[split_attr], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attr] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target_attr])
    return total_entropy - weighted_entropy

def id3(data, original_data, features, target_attr, parent_class=None):
    if len(np.unique(data[target_attr])) == 1:
        return np.unique(data[target_attr])[0]
    if len(data) == 0:
        return parent_class
    if len(features) == 0:
        return parent_class

    parent_class = np.unique(data[target_attr])[
        np.argmax(np.unique(data[target_attr], return_counts=True)[1])
    ]

    gains = [information_gain(data, feature, target_attr) for feature in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]
        subtree = id3(subset, original_data, remaining_features, target_attr, parent_class)
        tree[best_feature][value] = subtree

    return tree

def predict(sample, tree, default="Yes"):
    for key in sample:
        if key in tree:
            try:
                result = tree[key][sample[key]]
            except:
                return default
            if isinstance(result, dict):
                return predict(sample, result, default)
            else:
                return result

features = data.columns[:-1].tolist()
decision_tree = id3(data, data, features, "PlayTennis")

print(decision_tree)

new_sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

print(predict(new_sample, decision_tree))
