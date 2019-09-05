import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn import svm, tree   
import numpy as np


class XGBoostModel:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = None
        self.train_model = None
        self.Xtrain = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kfold = KFold(n_splits=10, shuffle=True)

    def xg_model(self, X_train, y_train):
        self.model = xgb.XGBClassifier(min_child_weight=20)
        self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        y_target = self.model.predict(X_test)
        return y_target

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        score = 0.0
        metrics = []
        for train_index, val_index in self.kfold.split(self.Xtrain):
            X_train, X_val, y_train, y_val = self.Xtrain.iloc[train_index], self.Xtrain.iloc[val_index], self.y_train.iloc[train_index], self.y_train.iloc[val_index]
            self.xg_model(X_train, y_train)  
            y_target = self.predict_model(X_val)
            tmp = self.accuracy(y_target, y_val)  
            if tmp > score:
                score = tmp
                metrics = [X_train, y_train]
        self.xg_model(metrics[0], metrics[1]) 
        y_target = self.predict_model(self.X_test)
        return self.accuracy(self.y_test, y_target)


class DecisionTreeModel:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = None
        self.train_model = None
        self.Xtrain = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kfold = KFold(n_splits=10, shuffle=True)

    def tree_model(self, X_train, y_train):
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        return self.model.predict(X_test) 

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        score = 0.0
        metrics = []
        for train_index, val_index in self.kfold.split(self.Xtrain):
            X_train, X_val, y_train, y_val = self.Xtrain.iloc[train_index], self.Xtrain.iloc[val_index], self.y_train.iloc[train_index], self.y_train.iloc[val_index]
            self.tree_model(X_train, y_train)  
            y_target = self.predict_model(X_val)
            tmp = self.accuracy(y_target, y_val)   
            if tmp > score:
                score = tmp
                metrics = [X_train, y_train]
        self.tree_model(metrics[0], metrics[1]) 
        y_target = self.predict_model(self.X_test)
        return self.accuracy(self.y_test, y_target)


class SVMModel:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = None
        self.train_model = None
        self.Xtrain = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        import numpy as np
        self.kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    def svm_model(self, X_train, y_train):
        self.model = svm.SVC(gamma='scale')
        self.model.fit(X_train, y_train)

    def predict_model(self, X_test):
        return self.model.predict(X_test) 

    def accuracy(self, y_test, y_target):
        return accuracy_score(y_test, y_target)

    def kfold_accuracy(self):
        score = 0.0
        metrics = []
        for train_index, val_index in self.kfold.split(self.Xtrain):
            X_train, X_val, y_train, y_val = self.Xtrain.iloc[train_index], self.Xtrain.iloc[val_index], self.y_train.iloc[train_index], self.y_train.iloc[val_index]
            self.svm_model(X_train, y_train)  
            y_target = self.predict_model(X_val)
            tmp = self.accuracy(y_target, y_val)   
            if tmp > score:
                score = tmp
                metrics = [X_train, y_train]
        self.svm_model(metrics[0], metrics[1]) 
        y_target = self.predict_model(self.X_test)
        return self.accuracy(self.y_test, y_target)
