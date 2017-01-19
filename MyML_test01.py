import pandas
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

testdata = pandas.read_csv("testdata.csv", sep="\t")
print(testdata.head(6))
print(testdata.describe())

testdata.loc[testdata["Gender"] == "F", "Gender"] = 0
testdata.loc[testdata["Gender"] == "M", "Gender"] = 1

testdata["NameLength"] = testdata["Name"].apply(lambda x: len(x))

print(testdata.head(6))
print(testdata.describe())

predictors = ["NameLength", "Ind1", "Ind2", "Gender"]
targets = ["point"]

alg = LinearRegression()

kf = KFold(n_splits=10, random_state=1)

predictions = []
for train, test in kf.split(testdata):
    # print("train: ", train, "test: ", test)
    train_predictors = testdata[predictors].iloc[train]
    train_target = testdata["point"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(testdata[predictors].iloc[test,:])
    predictions.append(test_predictions)
    print(test_predictions[0])
print(predictions)

# scores = cross_val_score(alg, testdata[predictors], testdata["point"], cv=kf.split(testdata))
#
# print(scores.mean())