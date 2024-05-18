import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing


# Define data viz class for visualizations for boston housing

class BostonVisualizer:
    def __init__(self, data, save_dir='artifacts/'):
            self.data = data
            self.save_dir = save_dir
 
    def save_plot(self, fig, filename):
        fig.savefig(f"{self.save_dir}{filename}")
        plt.close(fig)
        print(f"Plot saved as {filename}")

    def plot_boxplots(self, filename='boxplots.png'):
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
        axs = axs.flatten()
        for index, (k, v) in enumerate(self.data.items()):
            sns.boxplot(y=k, data=self.data, ax=axs[index])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        self.save_plot(fig, filename)

    def plot_regplots(self, filename = 'regplots.png'):
        min_max_scaler = preprocessing.MinMaxScaler()
        column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
        x = self.data.loc[:,column_sels]
        y = self.data['MEDV']
        x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
        fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for i, k in enumerate(column_sels):
            sns.regplot(y=y, x=x[k], ax=axs[i])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        self.save_plot(fig, filename)

def main():

    # define columns names, read in csv
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv('/Users/reymerekar/Desktop/ml_pipeline_airflow/data/boston_housing.csv', header=None, delimiter=r"\s+", names=column_names)

    # Create instance of class, create/save box plot and regplot using method
    visualizer = BostonVisualizer(data, save_dir='/Users/reymerekar/Desktop/ml_pipeline_airflow/artifacts/')
    visualizer.plot_boxplots('boxplot.png') 
    visualizer.plot_regplots('regplots.png')

if __name__ == "__main__":
    main()