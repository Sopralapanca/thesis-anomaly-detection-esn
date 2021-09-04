import pandas as pd
import numpy as np
folder = "LSTM_2layers_2021-08-31_09.27.38"
csv_path = "./data/{}/losses.csv".format(folder)

df = pd.read_csv(csv_path)
train_avg_list = []
valid_avg_list = []
valid_std_list = []
train_std_list = []

train_loss_df = df.filter(regex=("train_loss*"))
valid_loss_df = df.filter(regex=("valid_loss*"))

for i, row in train_loss_df.iterrows():
    train_avg = np.mean(list(row.values))
    train_avg_list.append(train_avg)

    train_std = np.std(list(row.values))
    train_std_list.append(train_std)

for i, row in valid_loss_df.iterrows():
    valid_avg = np.mean(list(row.values))
    valid_avg_list.append(valid_avg)

    valid_std = np.std(list(row.values))
    valid_std_list.append(valid_std)

df["Train AVG"] = train_avg_list
df["Valid AVG"] = valid_avg_list
df["Train STD"] = train_std_list
df["Valid STD"] = valid_std_list

df.to_csv(csv_path, sep=',', index=False)

#print della media