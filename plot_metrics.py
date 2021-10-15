import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging
from telemanom.helpers import Config
from telemanom.ROCcurve import roc_curve

config_path = "config.yaml"
config = Config(config_path)

parser = argparse.ArgumentParser(description='Parse path to p_values.csv.')
parser.add_argument('-p', '--path', default=None, required=True)
args = parser.parse_args()


def plotting_p(precision=None, recall=None, p=None, focus=False, run_id="",
               precision2=None, recall2=None, p2=None):
    fig, ax = plt.subplots()
    ax.scatter(precision, recall, s=150, label='p value')


    if focus:
        xoffset = 0.30
        switch = -0.6
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.scatter(precision2, recall2, s=180, color='red', label='best p values')
        j="2"
    else:
        xoffset = 0.1
        switch = -0.6

        bbox_props = None
        j="1"

    for i, txt in enumerate(p):
        if focus:
            try:

                if p[i] in p2.to_list():
                    txt = "p: " + str(txt) + "\nrecall: {0:.2f}\n".format(recall[i])
                    txt = txt + "precision: {0:.2f}".format(precision[i])

                    if precision[i] > 0.65:
                        position = precision[i] - 0.15
                    else:
                        position = precision[i]

                    ax.annotate(txt, (precision[i], recall[i]), size=15,
                                xytext=(position, recall[i] + switch * xoffset),
                                xycoords='data', textcoords='data',
                                bbox=bbox_props,
                                arrowprops=dict(arrowstyle="->", color="0.5",
                                                shrinkA=5, shrinkB=5,
                                                patchA=None, patchB=None,
                                                connectionstyle='arc3,rad=0.1',
                                                )

                                )

                    switch*=-1
                    xoffset *= 0.7
            except KeyError as e:
                pass
        else:
            try:
                if precision[i] > 0.75:
                    position = precision[i] - 0.05
                    xoffset *= 1.3
                else:
                    position = precision[i]
                ax.annotate(txt, (precision[i], recall[i]), size=16,
                        xytext=(position, recall[i] + switch * xoffset),
                        xycoords='data', textcoords='data',
                        bbox=bbox_props,
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle='arc3,rad=0.1',
                                        )

                         )
            except KeyError as e:
                pass

            switch *= -1

    ax.set_xlim(0.5, 0.8)
    ax.set_ylim(0.4, 1)
    ax.set_xlabel(r'Precision', fontsize=15)
    ax.set_ylabel(r'Recall', fontsize=15)
    ax.set_title('Precision vs Recall', fontsize=17)
    ax.legend(loc='best', shadow=True, fontsize='x-large')

    ax.grid(True)
    fig.set_size_inches(8, 6)

    save_path = run_id.replace("p_values.csv", "PrecisionVSRecall_{}.png".format(j))
    plt.savefig(save_path)
    plt.show()
    plt.close()



def metrics(path, log_path):
    df = pd.read_csv(path)

    # plot precision and recall graph for the various values of p
    sorted_df = df.sort_values(by='Recall', ascending=False)

    #taking the first 20 items
    n_values = 20

    p = sorted_df["P"].head(n_values)
    precision = sorted_df["Precision"].head(n_values)
    recall = sorted_df["Recall"].head(n_values)

    plotting_p(precision=precision, recall=recall, p=p, focus=False, run_id=path)

    # focus on best p values
    precision_threshold = config.precision_threshold
    recall_threshold = config.recall_threshold
    while True:

        subsetx = sorted_df[(sorted_df['Precision'] >= precision_threshold) & (sorted_df['Recall'] >= recall_threshold)]

        if not subsetx.empty:

            p2 = subsetx["P"]
            precision2 = subsetx["Precision"]
            recall2 = subsetx["Recall"]

            plotting_p(precision=precision, recall=recall, p=p, focus=True, run_id=path,
                       precision2=precision2, recall2=recall2, p2=p2)
            break
        else:
            with open(log_path, 'a') as fh:
                fh.write("No values found with threshold recall: {} precision: {}".format(recall_threshold, precision_threshold))
            recall_threshold -= 0.05

    # roc curve
    auc = roc_curve(path)

    with open(log_path, 'a') as fh:
        fh.write("AUC score: {}\n".format(auc))



if __name__ == '__main__':
    path = args.path
    try:
        with open(path, 'r') as fh:
            log_path = path.replace("data/", "data/logs/")
            log_path2 = log_path.replace("/p_values.csv", ".log")
        fh.close()
    # Load configuration file values
    except Exception as e:
        raise e

    metrics(path, log_path2)