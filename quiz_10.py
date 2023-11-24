import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix

filename = "./data/09_irisdata.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filename, names=names)

print("데이터 셋의 행렬 크기:", dataset.shape)

print("데이터 셋의 요약:")
print(dataset.describe())

print("데이터 셋의 클래스 종류 및 개수:")
print(dataset.groupby('class').size())

scatter_matrix(dataset)
plt.savefig('scatter_matrix.png')
plt.show()

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

model = DecisionTreeClassifier()

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')

print(results.mean())
