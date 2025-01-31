import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

knn = KNN(k=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)

acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)