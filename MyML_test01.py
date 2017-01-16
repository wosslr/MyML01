import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

testdata = pandas.read_csv("testdata.csv", sep="\t")
print(testdata.head(6))
print(testdata.describe())

testdata.loc[testdata["Gender"] == "F", "Gender"] = 0.5
testdata.loc[testdata["Gender"] == "M", "Gender"] = 0.6

print(testdata.head(6))
print(testdata.describe())

predictors = ["Name", "Ind1", "Ind2", "Gender"]

# alg = LinearRegression()
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

kf = KFold(testdata.shape[0], random_state=1)

# predictions = []
# for train, test in kf.split(testdata):
#     train_predicors = (testdata[predictors].iloc[train,:])
#     train_target = testdata["point"].iloc[train]
#     alg.fit(train_predicors, train_target)
#     test_predictions = alg.predict(testdata[predictors].iloc[test,:])
#     predictions.append(test_predictions)
# print(predictions)

# scores = cross_val_score(alg, testdata[predictors], testdata["point"], cv=kf.split(testdata))
#
# print(scores.mean())


print()