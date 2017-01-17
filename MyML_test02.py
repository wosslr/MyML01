import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

testdata = pandas.read_csv("testdata.csv", sep="\t")
# print(testdata.head(6))
# print(testdata.describe())
#
# testdata.loc[testdata["Gender"] == "F", "Gender"] = 0
# testdata.loc[testdata["Gender"] == "M", "Gender"] = 1
#
# print(testdata.head(6))
# print(testdata.describe())
#
# predictors = ["Name", "Ind1", "Ind2", "Gender"]
testdata["NameLength"] = testdata["Name"].apply(lambda x: len(x))
predictors = ["NameLength", "Ind1", "Ind2", "Gender"]

print(testdata[predictors].iloc[[3,5,6],:])

testdata2 = testdata.iloc[8:]

print(testdata2.loc["Name"])