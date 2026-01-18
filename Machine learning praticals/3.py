import pandas as pd
import math
# Load dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny',
                'Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
             'Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No',
                    'Yes','Yes','Yes','Yes','Yes','No']
}
df = pd.DataFrame(data)
# Entropy calculation
def entropy(col):
    values = col.value_counts()
    total = len(col)
    return sum([- (v/total) * math.log2(v/total) for v in values])
# Information Gain
def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    values = df[attr].unique()
    weighted_entropy = 0
    for v in values:
        subset = df[df[attr] == v]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy
# ID3 algorithm
def id3(df, target, attributes):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attributes:
        return df[target].mode()[0]
    gains = {attr: info_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    for value in df[best_attr].unique():
        subset = df[df[best_attr] == value]
        remaining_attrs = [a for a in attributes if a != best_attr]
        tree[best_attr][value] = id3(subset, target, remaining_attrs)
    return tree
# Build decision tree
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
decision_tree = id3(df, 'PlayTennis', attributes)
print("Decision Tree:")
print(decision_tree)
