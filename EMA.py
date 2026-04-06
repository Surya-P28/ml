# Program

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load data
iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=[
    'sepal length', 'sepal width', 'petal length', 'petal width'
])

y = iris.target

colors = np.array(['red', 'lime', 'black'])

# GMM preprocessing
xs = pd.DataFrame(preprocessing.StandardScaler().fit_transform(X),
                  columns=X.columns)

# KMeans
km = KMeans(n_clusters=3).fit(X)
km_labels = km.labels_

# GMM
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_gmm = gmm.predict(xs)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

for a, labels, title in zip(ax,
                            [y, km_labels, y_gmm],
                            ['Real', 'KMeans', 'GMM']):

    a.scatter(X['petal length'],
              X['petal width'],
              c=colors[labels],
              s=40)

    a.set_title(title)
    a.set_xlabel("Petal length")
    a.set_ylabel("Petal width")

plt.tight_layout()
plt.show()

# Results

print(f"KMeans Accuracy: {metrics.accuracy_score(y, km_labels)}")

print(f"KMeans Confusion Matrix:\n{metrics.confusion_matrix(y, km_labels)}")

print(f"GMM Accuracy: {metrics.accuracy_score(y, y_gmm)}")

print(f"GMM Confusion Matrix:\n{metrics.confusion_matrix(y, y_gmm)}")
