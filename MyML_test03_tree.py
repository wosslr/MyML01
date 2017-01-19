import pandas
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pydot

testdata = pandas.read_csv("test_logreg.csv", sep="\t")
print(testdata.head(6))
print(testdata.describe())

# pre-processing

print(testdata.head(6))
print(testdata.describe())

predictors = ["IND1", "IND2", "IND3"]
targets = ["RES"]

# alg = LogisticRegression(C=1e9)
# alg = LinearRegression()
alg = tree.DecisionTreeClassifier()

alg.fit(testdata[predictors], testdata[targets])

tree.export_graphviz(alg, out_file="tree.dot", feature_names=predictors, class_names=["False", "True"])

(graph,) = pydot.graph_from_dot_file("tree.dot")
graph.write_png("tree.png")

kf = KFold(n_splits=300, random_state=1)

# predictions = []
# for train, test in kf.split(testdata):
#     # print("train: ", train, "test: ", test)
#     train_predictors = testdata[predictors].iloc[train]
#     train_target = testdata["RES"].iloc[train]
#     print(train)
#     print(test)
#     alg.fit(train_predictors, train_target)
#     test_predictions = alg.predict(testdata[predictors].iloc[test,:])
#     predictions.append(test_predictions)
#     print(test_predictions[0])
# print(predictions)

# scores = cross_val_score(alg, testdata[predictors], testdata["point"], cv=kf.split(testdata))
#
# print(scores.mean())

while True:
    test_input = []
    ind1 = raw_input("IND1: ")
    ind2 = raw_input("IND2: ")
    ind3 = raw_input("IND3: ")
    test_input.append([float(ind1), float(ind2), float(ind3)])
    test_result = alg.predict(test_input)
    print("result: ")
    print(test_result)