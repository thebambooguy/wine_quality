from Model import LinearRegressionModel, SVM_Model, LogisticRegressionModel, RandomForestModel
from DbReader import DbReader
from sklearn.preprocessing import scale
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def reduce_class(array):
    array[array < 4] = 0
    array[array >= 7] = 2
    array[np.logical_and(array < 7, array >= 4)] = 1
    return array

def load_datasets(db_reader):
    redX, redy = db_reader.load_csv_and_create_dataset("winequality-red.csv")
    whiteX, whitey = db_reader.load_csv_and_create_dataset("winequality-white.csv")
    return redX, redy, whiteX, whitey

def split_dataset(red_X, white_X, red_y, white_y,n):
    X, y = db_reader.concatenate_data(red_X, white_X, red_y, white_y)

    X_for_training = X.head(int(len(X) * (n / 100)))
    y_for_training = y.head(int(len(y) * (n / 100)))

    X_test = X.tail(int(len(X) * ((100 - n) / 100)))
    y_test = y.tail(int(len(y) * ((100 - n) / 100)))

    return X_for_training, y_for_training, X_test, y_test

def train_script(training_X, training_y):

    models = [LinearRegressionModel(), SVM_Model(), LogisticRegressionModel(), RandomForestModel()]

    for model in models:

        fitting_model = model.gridsearchCV()
        fitting_model.fit(training_X, training_y)
        print(fitting_model.best_score_)
        model.set_internal_model(fitting_model.best_estimator_)
        model.save_model()

db_reader = DbReader()
red_X, red_y, white_X, white_y = load_datasets(db_reader)
X_train, y_train, X_test, y_test = split_dataset(red_X, white_X, red_y, white_y, 80)


feature_name_to_drop = ['fixed acidity','volatile acidity','citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates']
reduced_X_train = X_train.drop(feature_name_to_drop,axis=1)

scaled_training_X = scale(reduced_X_train)
#y_train = reduce_class(y_train)

if __name__ == "__main__":
    train_script(scaled_training_X,y_train)


