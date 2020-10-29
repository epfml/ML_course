"""ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Balance(y):
    """Plots the piechart of labels y"""
    count= sum([x>0 for x in y])
    labels = '1: Signal', '-1: Background'
    sizes = [count, len(y)-count]
    plt.pie(sizes, explode=(0, 0.1), labels=labels, autopct='%1.1f%%')
    plt.title('Pie chart of the labels in our dataset')
    plt.savefig("Target variable balance")
    plt.show()
    

def histogram(data, columns):
    """Plots the histogram of the features in columns."""
    fig, axes = plt.subplots(10, 3, figsize=(15,30))
    for ind, feature in enumerate(columns):
        row = ind % 10
        col = ind % 3
        sns.distplot(data[:,ind] , rug=True, ax=axes[row, col])
        axes[row, col].set_title("{}, feature nbr{}".format(columns[ind],ind))
    plt.tight_layout(pad=0.2)
    plt.savefig("Data distributions")
    plt.show()
    
    
def jet_histo(data, columns_names):
    nbr_features = len(columns_names)
    fig, axes = plt.subplots(nbr_features, figsize=(10,30))
    axs = axes.ravel()
    for ind, feature in enumerate(columns_names):
        sns.distplot(data[:,ind] , rug=True, ax=axs[ind])
        axs[ind].set_title("{}, feature nbr{}".format(columns_names[ind],ind))
    plt.tight_layout(pad=0.2)
    plt.savefig("Data distributions")
    plt.show()
    
def correlation_matrix(data, col_names):
    """Plot the correlation matrix of data"""
    nbr_cols= data.shape[1]
    plt.figure(figsize=(20, 10))
    corr = np.zeros((nbr_cols,nbr_cols))
    corr = np.corrcoef(data.T)        
    ax = sns.heatmap(corr, xticklabels=col_names, yticklabels=col_names, 
                     linewidths=.2, cmap="magma")
    plt.title('Correlation heatmap between features')
    plt.savefig("Correlation matrix heatmap")
    plt.show()