import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm, tree   
import numpy as np


class XGBoostModel:

    def __init__(self, X, y):
        self.model = None
        self.train_model = None
        self.X = X
        self.y = y
        self.kfold = KFold(n_splits=10, random_state=42, shuffle=True)

    def xg_model(self, X_train, y_train):
        self.model = xgb.XGBClassifier()
        self.train_model = self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        y_target = self.train_model.predict(X_test)
        return y_target

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        scores = []
        for train_index, test_index in self.kfold.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.xg_model(X_train, y_train)
            y_target = self.predict_model(X_test)
            scores.append(self.accuracy(y_test, y_target))
        return np.mean(scores)


class DecisionTreeModel:

    def __init__(self, X, y):
        self.model = None
        self.train_model = None
        self.X = X
        self.y = y
        self.kfold = KFold(n_splits=10, random_state=42, shuffle=True)

    def tree_model(self, X_train, y_train):
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        return self.model.predict(X_test) 

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        scores = []
        for train_index, test_index in self.kfold.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.tree_model(X_train, y_train)
            y_target = self.predict_model(X_test)
            scores.append(self.accuracy(y_test, y_target))
        return np.mean(scores)


class SVMModel:

    def __init__(self, X, y):
        self.model = None
        self.train_model = None
        self.X = X
        self.y = y
        self.kfold = KFold(n_splits=10, random_state=42, shuffle=True)

    def svm_model(self, X_train, y_train):
        self.model = svm.SVC(gamma='scale')
        self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        return self.model.predict(X_test) 

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        scores = []
        for train_index, test_index in self.kfold.split(self.X):
            X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[test_index]
            self.svm_model(X_train, y_train)
            y_target = self.predict_model(X_test)
            scores.append(self.accuracy(y_test, y_target))
        return np.mean(scores)
