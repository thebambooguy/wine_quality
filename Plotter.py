import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:

    def __init__(self, X= None, y = None):
        self.X = X
        self.y = y


    def plot_heatmap(self, X, y):
        X['quality'] = y
        sns.heatmap(X.corr(), annot=True)
        plt.show()

    def plot_pairplot(self, X, y):
        features = ['alcohol']
        X['quality'] = y
        sns.pairplot(X, vars=features,hue='quality')
        plt.show()

    def plot_dependecies(self, X, y):

        for feature_name in list(X.columns.values):
            plt.figure(figsize=(4,4))
            sns.scatterplot(x = X[feature_name], y= y, hue= y)
        plt.show()