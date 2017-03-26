from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=.5)

# from sklearn import tree
# cls = tree.DecisionTreeClassifier()

from sklearn import neighbors
cls = neighbors.KNeighborsClassifier()

cls.fit(X_train, Y_train)

predictions = cls.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))