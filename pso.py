from pre_process import *
from models import *
# Import modules
import numpy as np
import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, 
                                     plot_surface)
# from pyswarms.utils.plotters.formatters import Mesher
# from pyswarms.utils.plotters.formatters import Designer

class PSO:

    def __init__(self, path='../data/pokemon.csv', label_name='target', mtype='xgboost'):
        self.pre_processing = PreProcessing(path)
        self.data = self.pre_processing.load_dataset()
        self.x, self.y = self.pre_processing.separate_data(self.data, label_name)
        self.label_name = label_name
        self.optimizer = None
        self.mtype = mtype

    def convert_particle(self, x):
        for particle in x:
            for pos in range(0, len(particle)):
                if particle[pos] < 0.5: particle[pos] = 0
                else: particle[pos] = 1
        return x

    def define_columns(self, x):
        new_data = self.pre_processing.copy_columns(x, self.label_name)
        X_train, X_test, y_train, y_test = self.pre_processing.split_train_test(new_data, self.label_name)
        return X_train, X_test, y_train, y_test

    def model_accuracy(self, x):
        res = []
        if self.mtype == 'xgboost':
            res = self.model_accuracy_xgboost(x)
        if self.mtype == 'svm':
            res = self.model_accuracy_svm(x)
        if self.mtype == 'tree':
            res = self.model_accuracy_tree(x)
        return res

    def model_accuracy_xgboost(self, x):
        x = self.convert_particle(x)
        res = []
        n_particles = x.shape[0]
        for particle in range(0, n_particles):
            # X_train, X_test, y_train, y_test = self.define_columns(x[particle])
            xgbmodel = XGBoostModel(self.x, self.y)
            # xgbmodel.xg_model()
            # y_pred = xgbmodel.predict_model()
            res.append(-xgbmodel.kfold_accuracy())
        return np.array(res)

    def model_accuracy_svm(self, x):
        x = self.convert_particle(x)
        res = []
        n_particles = x.shape[0]
        for particle in range(0, n_particles):
            # X_train, X_test, y_train, y_test = self.define_columns(x[particle])
            svmmodel = SVMModel(self.x, self.y)
            # svmmodel.svm_model()
            # y_pred = svmmodel.predict_model()
            res.append(-svmmodel.kfold_accuracy())
        return np.array(res)

    def model_accuracy_tree(self, x):
        x = self.convert_particle(x)
        res = []
        n_particles = x.shape[0]
        for particle in range(0, n_particles):
            # X_train, X_test, y_train, y_test = self.define_columns(x[particle])
            treemodel = DecisionTreeModel(self.x, self.y)
            # treemodel.tree_model()
            # y_pred = treemodel.predict_model()
            res.append(-treemodel.kfold_accuracy())
        return np.array(res)

    def optimize_pso(self):
        max_bound = 1 * np.ones(len(self.data.columns))
        min_bound = 0 * np.ones(len(self.data.columns))
        limit = (min_bound, max_bound)
        # instatiate the optimizer(parameters for the swarms(weight to the communication of the swarms(best swarms, best local swarms and inertia)))
        options = {'c1': 0.1, 'c2': 0.3, 'w': 1}
        # #Create swarm with n_particles, dimensions of dimensional space, parameters for the swarms and bounds
        self.optimizer = ps.single.GlobalBestPSO(n_particles=3, dimensions=len(self.data.columns), options=options, bounds=limit)
        cost, pos = self.optimizer.optimize(self.model_accuracy, iters=10)
        return cost, pos

    def plot_cost(self):
        plot_cost_history(cost_history=self.optimizer.cost_history)
        plt.show()