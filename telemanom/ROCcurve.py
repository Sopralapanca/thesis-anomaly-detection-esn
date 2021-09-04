import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)


def roc_curve(id):
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


    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr_list, tpr_list)
    plt.savefig('./data/{}/roc_curve.png'.format(id))
    plt.show()

    return(auc(fpr_list,tpr_list))

