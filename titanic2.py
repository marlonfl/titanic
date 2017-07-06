import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

if __name__ == "__main__":

    # loading data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # save for correct output file formatting
    p_id = test.loc[:, 'PassengerId']

    # saving index where training data ends
    train_index = train.shape[0]

    # saving targets before combining
    y = train.Survived
    train.drop(["Survived"], axis=1, inplace=True)

    # combining train and test data so data transformations are applied to both
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    # mapping binary feature to 1/0
    combined.Sex.replace(['male', 'female'], [1, 0], inplace=True)

    # fixing Age NaNs
    # rather that picking just the median age, I'm trying to assign even better ages
    # train.groupby(['Sex', 'Pclass'])['Age'].median()
    # ------>
    # Sex  Pclass
    # 0    1         35.0
    #      2         28.0
    #      3         21.5
    # 1    1         40.0
    #      2         30.0
    #      3         25.0
    # Name: Age, dtype: float64

    train_age_median = combined.head(891).Age.median()
    test_age_median = combined.iloc[891:].Age.median()
    combined.Age.loc[:891][np.isnan(combined.Age.loc[:891])] = train_age_median
    combined.Age.loc[891:][np.isnan(combined.Age.loc[891:])] = test_age_median

    # - Dropping useless PassengedId column, has no info for classification
    # - Also dropping Ticket feature because it probably carries little information
    #   and is too annoying to deal with
    # - Cabin might have some information but is annoying too and mostly NaNs
    combined.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Embarked has a few NaNs, filling it with the most common value S
    combined["Embarked"].fillna('S', inplace=True)

    # Embarked is a categorical feature with no order to the categories so it should be dummy encoded
    embarked_dummies = pd.get_dummies(combined["Embarked"], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop(['Embarked'], axis=1, inplace=True)

    # generating feature Alone
    # is using information from the existing SibSp/Parch features
    # 0 if the person has family aboard, 1 if not
    alone = [1 if combined["Parch"][i] + combined["SibSp"][i] == 0 else 0 for i in range(combined.shape[0])]
    combined["Alone"] = alone
    #combined.drop(['Parch', 'SibSp'], axis=1, inplace=True)

    # generating feature Titles, which are extracted from the subjects' names
    combined["Titles"] = [name.split(", ")[1].split(".")[0] for name in combined["Name"]]
    combined.drop(['Name'], axis=1, inplace=True)

    # filtering out uncommon titles and/or titles that carry littly information
    combined["Titles"] = combined["Titles"].apply(lambda x : "NaN" if x not in ["Dr", "Miss", "Mrs", "Master"] else x)

    # titles is a categorical variable without ordner so dummy encoding
    title_dummies = pd.get_dummies(combined["Titles"], prefix='Title')
    title_dummies.drop(["Title_NaN"], axis=1, inplace=True)
    combined = pd.concat([combined,title_dummies], axis=1)
    combined.drop(["Titles"], axis=1, inplace=True)

    # Fare has NaN values, just inserting median
    train_fare_median = combined.head(891).Fare.median()
    test_fare_median = combined.iloc[891:].Fare.median()
    combined.Fare.loc[:891][np.isnan(combined.Fare.loc[:891])] = train_fare_median
    combined.Fare.loc[891:][np.isnan(combined.Fare.loc[891:])] = test_fare_median

    # normalizing
    #combined = (combined - combined.min()) / (combined.max() - combined.min())

    #
    # Data Preprocessing/Feature Engineering concluded
    #

    X_train = combined.head(891)
    y_train = y
    #X_test = combined.iloc[891:]

    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(X_train, y_train)

    features = pd.DataFrame()
    features['feature'] = X_train.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=False, inplace=True)
    features.set_index('feature', inplace=True)

    features.plot(kind="barh")
    plt.show()

    print(X_train.corr())
    #print (X_test[X_test.isnull()])
    #X_small_train, X_val, y_small_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5)

    #kNN = KNeighborsClassifier(3)
    #kNN.fit(X_small_train, y_small_train)

    #Cs = [0.001, 0.01, 0.1, 0.5, 1, 2, 4, 8, 9, 10, 11, 12, 16, 20]
    #gammas = [0.001, 0.01, 0.03, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3]
    #param_grid = {'C': Cs, 'gamma' : gammas}
    #grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
#    grid_search.fit(X_small_train, y_small_train)
    #p = grid_search.best_params_

    #p = {"gamma":0.25, "C":10}
    #s = svm.SVC(gamma=p["gamma"], C=p["C"])
    #s.fit(X_small_train, y_small_train)
    #svmpred = s.predict(X_val)
    #print ("SVM: " + str(accuracy_score(y_val, svmpred)))

    #forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=3)
    #forest.fit(X_small_train, y_small_train)

    #svma = svm.SVC()
    #svma.fit(X_small_train, y_small_train)

    #nb = GaussianNB()
    #nb.fit(X_small_train, y_small_train)

    #y_val_preds = nb.predict(X_val)

    #knn_real = KNeighborsClassifier(3)
    #knn_real.fit(X_train, y_train)
    #y_pred = knn_real.predict(X_test)

    #d = {'PassengerId':p_id, 'Survived':y_pred}
    #s = pd.DataFrame(d)
    #s.to_csv("pred_svm.csv", index=False)

    print ()
