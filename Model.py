from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from abc import ABC
import numpy as np

class Model(ABC):
    def __init__(self, X_train = None, y_train = None, data_to_scale = None, model_filename = None):
        self.X_train = X_train
        self.y_train = y_train
        self.data_to_scale = data_to_scale
        self.model_filename = model_filename

    def fit(self, X_train , y_train):
        self.internal_model.fit(X_train, y_train)

    def predict(self, X_train):
        temp = self.internal_model.predict(X_train)
        return temp

    def scale(self, data_to_scale):
        return scale(data_to_scale)

    def gridsearchCV(self):
        best_model = GridSearchCV()
        return best_model

    def save_model(self):
        dump(self, self.model_filename)

    def load_model(self):
        self.model = load(self.model_filename)
        return self.model

    def set_internal_model(self, external_model):
        self.internal_model = external_model


class LinearRegressionModel(Model):
    def __init__(self, fit_intercept = True, normalize = False, copy_X = True, n_jobs = None):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.internal_model = LinearRegression(fit_intercept=self.fit_intercept,normalize= self.normalize,copy_X=self.copy_X,n_jobs=self.n_jobs)
        self.model_filename = "LinearRegressionModelparameters.joblib"

        self.parameters = {'fit_intercept': [True, False], 'normalize': [True, False]}
        self.cv = 10
        self.scoring = 'neg_mean_squared_error'

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model,self.parameters,cv= self.cv, scoring=self.scoring)
        return best_model

    def predict(self, X_train):
        temp = self.internal_model.predict(X_train)
        return np.around(temp)

class SVM_Model(Model):
    def __init__(self,C=1, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=1e-3, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.internal_model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0 = self.coef0, shrinking=self.shrinking, probability=self.probability, tol=self.tol, verbose=self.verbose, max_iter = self.max_iter, decision_function_shape =self.decision_function_shape, random_state = self.random_state)
        self.model_filename = "SVMModelparameters.joblib"

        #self.parameters = {'C': [1.0, 3.0, 5.0, 10.0, 20.0], 'kernel': ['rbf'], 'degree': [3, 5, 7], 'class_weight': [{0: 5., 1: 0.5, 2: 5}]}
        self.parameters = {'C': [1.0, 3.0, 5.0, 10.0, 20.0], 'kernel': ['rbf'], 'degree': [3, 5, 7]}
        self.cv = 10
        self.scoring = 'balanced_accuracy'

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model,self.parameters,cv= self.cv, scoring=self.scoring)
        return best_model


class LogisticRegressionModel(Model):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state= None, max_iter=100, solver = 'newton-cg'):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.solver = solver
        self.internal_model = LogisticRegression(penalty=self.penalty, dual=self.dual, tol=self.tol, C=self.C, fit_intercept=self.fit_intercept, intercept_scaling=self.intercept_scaling, random_state=self.random_state, max_iter=self.max_iter)
        self.model_filename = "LogisticRegressionModelparameters.joblib"

        #self.parameters = {'penalty': ['l2'], 'dual': [True, False], 'C': [1.0, 3.0, 5.0, 15.0, 40.0], 'fit_intercept': [True, False], 'max_iter': [100,500,1000],  'class_weight': [{0: 5., 1: 0.5, 2: 5}]}
        self.parameters = {'penalty': ['l2'], 'dual': [True, False], 'C': [1.0, 3.0, 5.0, 15.0, 40.0],'fit_intercept': [True, False], 'max_iter': [100, 500, 1000]}
        self.cv = 10
        self.scoring = 'balanced_accuracy'

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model,self.parameters,cv= self.cv, scoring=self.scoring)
        return best_model

class RandomForestModel(Model):
    def __init__(self, n_estimators=5, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto", max_leaf_nodes=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.internal_model = RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,max_leaf_nodes=self.max_leaf_nodes,random_state=self.random_state)
        self.model_filename = "RandomForestModelparameters.joblib"

        #self.parameters = {'n_estimators': [10, 50, 100, 150, 200, 15 ], 'max_depth': [None, 2, 5, 10, 6], 'max_leaf_nodes': [None, 2, 5],  'class_weight': [{0: 5., 1: 0.5, 2: 5}]}
        self.parameters = {'n_estimators': [10, 50, 100, 150, 200, 15], 'max_depth': [None, 2, 5, 10, 6], 'max_leaf_nodes': [None, 2, 5]}
        self.cv = 10
        self.scoring = 'balanced_accuracy'

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model,self.parameters,cv= self.cv, scoring=self.scoring)
        return best_model


