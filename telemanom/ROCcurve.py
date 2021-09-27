import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc



def plot_roc_curve(fpr, tpr):
    """
    Plot the Receiver Operating Characteristic curve
    :param fpr: list containing the values of the false positive rate as p varies
    :param tpr: list containing the values of the true positive rate as p varies
    :return: plot the Receiver Operating Characteristic curve
    """

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')  # dashed diagonal

    ax.set_xlim(0.0, 0.81)
    ax.set_ylim(0.0, 0.81)
    ax.set_xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=16)

    ax.grid(True)

    fig.set_size_inches(8, 6)

    plt.savefig('./data/{}/roc_curve.png'.format(id))
    plt.show()


def roc_curve(id):
    """
    Given the name of a folder of an experiment, it creates the graph for the roc curve and calculates the auroc score
    :param id: folder name
    :return: auroc score
    """


    path='./data/{}/p_values.csv'.format(id)
    df = pd.read_csv(path)

    p = []
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    #itero sul file f_values.csv
    for index, row in df.iterrows():
        p.append(row["P"])
        true_positive_rate = row["Recall"]
        recall_list.append(row["Recall"])
        precision_list.append(row['Precision'])
        tpr_list.append(true_positive_rate)
        fp = row["Total False Positives"]
        tn = row["Total True Negatives"]
        false_positive_rate = fp / (fp + tn)
        fpr_list.append(false_positive_rate)


    plot_roc_curve(fpr_list, tpr_list)


    return(auc(fpr_list,tpr_list))

