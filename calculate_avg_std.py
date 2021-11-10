import pandas as pd
import numpy as np
import os.path
import argparse
import logging


parser = argparse.ArgumentParser(description='Parse name to project folder.')
parser.add_argument('-f', '--folder_name', default=None, required=True)
args = parser.parse_args()

def calculate_avg_std(folder):
    """
    :param folder: folder name to which the losses.csv file belongs
    :return: if the file exists, it calculates the mean and standard deviation and writes them to the file
    """
    log_path = "./data/logs/{}.log".format(folder)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    csv_path = "./data/{}/losses.csv".format(folder)
    if os.path.isfile(csv_path):
        print("losses file found at {}".format(csv_path))
    else:
        print("Can't find losses file at {}. Start the program with \"-f folder_name\" for example \"-f ESN_1layer_2021-08-31_20.45.38\"".format(csv_path))
        return

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

    valid_avg_column = df["Valid AVG"]
    valid_avg_max_value = valid_avg_column.max()
    valid_std_column = df["Valid STD"]
    valid_std_max_value = valid_std_column.max()

    logging.info("highest average loss on the validation set: {}".format(valid_avg_max_value))
    logging.info("highest standard deviation on the validation set: {}".format(valid_std_max_value))



if __name__ == '__main__':
    calculate_avg_std(folder=args.folder_name)
