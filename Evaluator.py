from sklearn.metrics import classification_report
from Model import LinearRegressionModel, SVM_Model, LogisticRegressionModel, RandomForestModel
from Training_Script import load_datasets, split_dataset, reduce_class
from sklearn.preprocessing import scale
from DbReader import DbReader
from Plotter import Plotter


def evaluating_script(test_X, test_y):

    models = [LinearRegressionModel(), SVM_Model(), LogisticRegressionModel(), RandomForestModel()]

    for model in models:

        load_model = model.load_model()
        predicted_values = load_model.predict(test_X)

        print(model.__class__.__name__)
        print(classification_report(test_y, predicted_values))



db_reader = DbReader()
red_X, red_y, white_X, white_y = load_datasets(db_reader)
X_train, y_train, X_test, y_test = split_dataset(red_X, white_X, red_y, white_y, 80)

feature_name_to_drop = ['fixed acidity','volatile acidity','citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates']

reduced_X_test = X_test.drop(feature_name_to_drop, axis=1)
reduced_X_train = X_train.drop(feature_name_to_drop, axis=1)

scaled_test_X = scale(reduced_X_test)
scaled_training_X = scale(reduced_X_train)

#y_test = reduce_class(y_test)
#y_train = reduce_class(y_train)

if __name__ == "__main__":
    #evaluating_script(scaled_training_X, y_train)
    evaluating_script(scaled_test_X,y_test)

