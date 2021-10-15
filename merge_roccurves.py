import pandas as pd
import matplotlib.pyplot as plt

df_esn1l = pd.read_csv("./data/prova/p_values.csv")
df_esn2l = pd.read_csv("./data/LSTM_1layer_2021-08-28_09.27.38/p_values.csv")
"""
df_lstm1l = pd.read_csv("/content/gdrive/My Drive/telemanom/p_values_LSTM1L.csv")
df_lstm2l = pd.read_csv("/content/gdrive/My Drive/telemanom/p_values_LSTM2L.csv")
"""
df_list = [df_esn1l, df_esn2l]
fig, ax = plt.subplots()

for elem in df_list:
    p = []
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    # itero sul file f_values.csv
    for index, row in elem.iterrows():
        p.append(row["P"])
        true_positive_rate = row["Recall"]
        recall_list.append(row["Recall"])
        precision_list.append(row['Precision'])
        tpr_list.append(true_positive_rate)
        fp = row["Total False Positives"]
        tn = row["Total True Negatives"]
        false_positive_rate = fp / (fp + tn)
        fpr_list.append(false_positive_rate)

    ax.plot(fpr_list, tpr_list, linewidth=2)

    ax.set_xlim(0.0, 1)
    ax.set_ylim(0.0, 1)
    ax.set_xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=16)

ax.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
ax.grid(True)
ax.legend(('ESN_1L', 'ESN_2L', 'LSTM_1L', 'LSTM_2L', 'Random'), loc='upper right', shadow=True)
fig.set_size_inches(8, 6)
plt.show()