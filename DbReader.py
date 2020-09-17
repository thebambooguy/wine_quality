import pandas as pd

class DbReader:
    def __init__(self, filename=None,data_from_csv=None, target_values = None, features=None, n = None, X = None, y = None):
        self.filename = filename
        self.data_from_csv = data_from_csv
        self.target_values = target_values
        self.features = features
        self.n = n
        self.X = X
        self.y = y

    def load_csv_and_create_dataset(self,filename):
        self.data_from_csv = pd.read_csv(filename, sep=';')
        self.target_values = self.data_from_csv['quality']
        self.features = self.data_from_csv.drop('quality',axis=1)

        return self.features, self.target_values

    def concatenate_data(self, first_X, second_X, first_y, second_y):
        self.X = pd.concat([first_X, second_X])
        self.y = pd.concat([first_y, second_y])

        return self.X, self.y