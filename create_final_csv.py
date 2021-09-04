import pandas as pd

columns = ["Model", "Precision", "Recall", "True Positives", "False Positives", "False Negatives", "Training Time", "Execution Time","Max Params"]

lstm1_row = ["LSTM 1layer", "0.77", "0.64", "67", "20", "38", "0h:20m", "0h:26m", "10.118"]
esn1_row = ["ESN 1layer", "0.76", "0.65", "68", "22", "37", "1h:32m", "2h:01m", "10.000"]
lstm2_row = ["LSTM 2layers", "0.68", "0.75", "79", "37", "26", "0h:37m","0h:45m",  "10.460"]
esn2_row = ["ESN 2layers", "0.82", "0.72", "76", "17", "29", "2h:32m:","3h:38m", "10.000"]

rows = [lstm1_row, esn1_row, lstm2_row, esn2_row]

df = pd.DataFrame(columns=columns)
for i in range(len(rows)):
    df.loc[i] = rows[i]

df.to_csv('./data/final.csv', sep=',')