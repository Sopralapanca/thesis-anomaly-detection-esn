import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging
import csv
import time

from telemanom.helpers import Config
from telemanom.errors import Errors
import telemanom.helpers as helpers
from telemanom.channel import Channel
from telemanom.modeling import Model
from telemanom.find_hp import FindHP
from pathlib import Path
from functools import reduce

logger = helpers.setup_logging()

def secondsToStr(t):
    return "%dh:%02dm:%02ds.%d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])

def write_csv(first_part_path, row, col_header):
    path = first_part_path+"_ms.csv"
    csv_file = Path(path)
    if csv_file.is_file():
        with open(path, 'a', newline='') as filedata:
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
            writer.writerow(row)
    else:
        with open(path, 'a', newline='') as filedata:
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
            writer.writeheader()
            writer.writerow(row)

    for key in row:
        if key == "model":
            continue
        else:
            row[key] = secondsToStr(float(row[key]))

    path = first_part_path+".csv"
    csv_file = Path(path)
    if csv_file.is_file():
        with open(path, 'a', newline='') as filedata:
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
            writer.writerow(row)
    else:
        with open(path, 'a', newline='') as filedata:
            writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
            writer.writeheader()
            writer.writerow(row)

class Detector:
    def __init__(self, labels_path=None, result_path='results/',
                 config_path='config.yaml'):
        """
        Top-level class for running anomaly detection over a group of channels
        with values stored in .npy files. Also evaluates performance against a
        set of labels if provided.

        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml

        Attributes:
            labels_path (str): see Args
            results (list of dicts): holds dicts of results for each channel
            result_df (dataframe): results converted to pandas dataframe
            chan_df (dataframe): holds all channel information from labels .csv
            result_tracker (dict): if labels provided, holds results throughout
                processing for logging
            config (obj):  Channel class object containing train/test data
                for X,y for a single channel
            y_hat (arr): predicted channel values
            id (str): datetime id for tracking different runs
            result_path (str): see Args
        """

        self.labels_path = labels_path
        self.results = []
        self.result_df = None
        self.chan_df = None

        self.precision = None
        self.recall = None

        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }

        self.config = Config(config_path)
        self.y_hat = None

        # trains new models
        if self.config.execution == "train" or self.config.execution == "train_and_predict":
            if self.config.use_id != "":
                self.id = self.config.use_id
            else:
                architecture = self.config.model_architecture
                name = self.config.name
                self.id = "{}_{}_{}".format(architecture, name, dt.now().strftime('%Y-%m-%d_%H.%M.%S'))

        # load existing models or predictions
        if self.config.execution == "predict" or self.config.execution == "search_p":
            self.id = self.config.use_id


        if self.config.execution == "find_hp":
            architecture = self.config.model_architecture
            name = self.config.name
            self.id = "{}_{}_{}".format(architecture, name, dt.now().strftime('%Y-%m-%d_%H.%M.%S'))

        helpers.make_dirs(self.id)

        # add logging FileHandler based on ID
        hdlr = logging.FileHandler('data/logs/%s.log' % self.id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        self.result_path = result_path

        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = [x.split('.')[0] for x in os.listdir('data/test/')]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})

        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))


    def evaluate_sequences(self, errors, label_row):
        """
        Compare identified anomalous sequences with labeled anomalous sequences.

        Args:
            errors (obj): Errors class object containing detected anomaly
                sequences for a channel
            label_row (pandas Series): Contains labels and true anomaly details
                for a channel

        Returns:
            result_row (dict): anomaly detection accuracy and results
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores

        idx = label_row['anomaly_sequences']
        if idx[0][0] > 0 and idx[-1][-1] < label_row['num_values']:
            total_negatives = len(label_row['anomaly_sequences']) +1

        elif  idx[0][0] > 0:
            total_negatives = len(label_row['anomaly_sequences'])
        elif idx[-1][-1] < label_row['num_values']:
            total_negatives = len(label_row['anomaly_sequences'])
        else:
            total_negatives = len(label_row['anomaly_sequences']) -1



        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']

        else:
            true_indices_grouped = [list(range(e[0], e[1]+1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:

                    result_row['tp_sequences'].append(e_seq)

                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1

                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'],
                                                          matched_true_seqs, axis=0))


        # true negatives detected = true negatives - false positives

        n_fp = len(result_row['fp_sequences'])
        row_true_negative = total_negatives - n_fp
        if row_true_negative <0:
            row_true_negative = 0
        result_row['true_negatives'] = row_true_negative


        if self.config.execution != "search_p":
            logger.info('Channel Stats: TP: {}  FP: {}  FN: {} TN: {}'.format(result_row['true_positives'],
                                                                   result_row['false_positives'],
                                                                   result_row['false_negatives'],
                                                                   result_row['true_negatives']))

        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row

    def log_final_stats(self):
        """
        Log final stats at end of experiment.
        """
        if self.config.execution == "search_p":
            logger.info('P value: {}'.format(self.config.p))

        if self.labels_path:

            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('True Positives: {}'
                        .format(self.result_tracker['true_positives']))
            logger.info('False Positives: {}'
                        .format(self.result_tracker['false_positives']))
            logger.info('False Negatives: {}'
                        .format(self.result_tracker['false_negatives']))
            logger.info('True Negatives: {}\n'
                        .format(self.result_tracker['true_negatives']))
            try:
                self.precision = float(self.result_tracker['true_positives']) / (float(self.result_tracker['true_positives'] + self.result_tracker['false_positives']))
                self.recall = float(self.result_tracker['true_positives']) / (float(self.result_tracker['true_positives'] + self.result_tracker['false_negatives']))

                logger.info('Precision: {0:.2f}'.format(self.precision))
                logger.info('Recall: {0:.2f}\n'.format(self.recall))
            except ZeroDivisionError:

                logger.info('Precision: NaN')
                logger.info('Recall: NaN\n')

        else:
            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('Total channel sets evaluated: {}'
                        .format(len(self.result_df)))
            logger.info('Total anomalies found: {}'
                        .format(self.result_df['n_predicted_anoms'].sum()))
            logger.info('Avg normalized prediction error: {}'
                        .format(self.result_df['normalized_pred_error'].mean()))
            logger.info('Total number of values evaluated: {}\n'
                        .format(self.result_df['num_test_values'].sum()))


    def execute_detection(self):
        col_header = ["model", "total_elapsed_time"]
        training_times_row = {
            'model': self.id,
            'total_elapsed_time': 0
        }

        execution_times_row = {
            'model': self.id,
            'total_elapsed_time': 0
        }


        training_total_elapsed_time = 0
        training_start_time = 0
        training_end_time = 0

        execution_total_elapsed_time = 0
        execution_start_time = 0
        execution_end_time = 0

        if self.config.execution == "train":
            filename = "./data/{}/losses.csv".format(self.id)
            if os.path.isfile(filename):
                losses_df = pd.read_csv(filename)
            else:
                losses_df = pd.DataFrame()
                losses_df["Channel"] = self.chan_df["chan_id"]

            train_loss = []
            validation_loss = []

            path = './data/{}/models/seeds.log'.format(self.id)
            seed_file = Path(path)
            if seed_file.is_file():
                os.remove(path)




        for i, row in self.chan_df.iterrows():
            if self.config.execution != "search_p" and self.config.execution != "find_hp":
                logger.info('Stream # {}: {}'.format(i + 1, row.chan_id))


            channel = Channel(self.config, row.chan_id)
            channel.load_data()

            if self.config.execution == "train":
                model = Model(self.config, self.id, channel)
                train_loss.append(model.history.history["loss"][-1])
                validation_loss.append(model.history.history["val_loss"][-1])
                continue

            elif self.config.execution == "find_hp":
                training_start_time = time.time()
                tuner = FindHP(self.id, channel, self.config, self.config.model_architecture)
                training_end_time = (time.time() - training_start_time)


                f = open('hp/{}/times.log'.format(self.id), "a")
                f.write("{} {}\n".format(row.chan_id, training_end_time))
                f.close()


                col_header.append(row.chan_id)
                training_times_row[row.chan_id] = training_end_time

                training_total_elapsed_time += training_end_time

                training_start_time = 0
                training_end_time = 0
                continue

            elif self.config.execution == "train_and_predict" or self.config.execution == "predict":
                if self.config.execution == "train_and_predict":
                    training_start_time = time.time()
                    execution_start_time = training_start_time

                model = Model(self.config, self.id, channel)

                if self.config.execution == "train_and_predict":
                    training_end_time = (time.time() - training_start_time)

                    col_header.append(row.chan_id)
                    training_times_row[row.chan_id] = training_end_time

                    training_total_elapsed_time += training_end_time

                    training_start_time = 0
                    training_end_time = 0

                channel = model.batch_predict(channel)

            elif self.config.execution == "search_p":
                channel.y_hat = np.load(os.path.join('data', self.id, 'y_hat',
                                                     '{}.npy'
                                                     .format(channel.id)))


            errors = Errors(channel, self.config, self.id)
            errors.process_batches(channel)
            if self.config.execution == "search_p":
                result_row = {
                    'run_id': self.id+"_{}".format(str(self.config.p)),
                    'chan_id': row.chan_id,
                    'num_train_values': len(channel.X_train)+len(channel.X_valid),
                    'num_test_values': len(channel.X_test),
                    'n_predicted_anoms': len(errors.E_seq),
                    'normalized_pred_error': errors.normalized,
                    'anom_scores': errors.anom_scores
                }
            else:
                result_row = {
                    'run_id': self.id,
                    'chan_id': row.chan_id,
                    'num_train_values': len(channel.X_train) + len(channel.X_valid),
                    'num_test_values': len(channel.X_test),
                    'n_predicted_anoms': len(errors.E_seq),
                    'normalized_pred_error': errors.normalized,
                    'anom_scores': errors.anom_scores
                }

            if self.labels_path:
                result_row = {**result_row,
                              **self.evaluate_sequences(errors, row)}
                result_row['spacecraft'] = row['spacecraft']
                result_row['anomaly_sequences'] = row['anomaly_sequences']
                result_row['class'] = row['class']
                self.results.append(result_row)

                if self.config.execution != "search_p":
                    logger.info('Total true positives: {}'
                                .format(self.result_tracker['true_positives']))
                    logger.info('Total false positives: {}'
                                .format(self.result_tracker['false_positives']))
                    logger.info('Total false negatives: {}'
                                .format(self.result_tracker['false_negatives']))
                    logger.info('Total true negatives: {}\n'
                                .format(self.result_tracker['true_negatives']))

            else:
                result_row['anomaly_sequences'] = errors.E_seq
                self.results.append(result_row)

                if self.config.execution != "search_p":
                    logger.info('{} anomalies found'
                                .format(result_row['n_predicted_anoms']))
                    logger.info('anomaly sequences start/end indices: {}'
                                .format(result_row['anomaly_sequences']))
                    logger.info('number of test values: {}'
                                .format(result_row['num_test_values']))
                    logger.info('anomaly scores: {}\n'
                                .format(result_row['anom_scores']))

            if self.config.execution != "search_p":
                self.result_df = pd.DataFrame(self.results)
                self.result_df.to_csv(
                    os.path.join(self.result_path, '{}.csv'.format(self.id)),
                    index=False)

            if self.config.execution == "train_and_predict":
                execution_end_time = (time.time() - execution_start_time)
                execution_times_row[row.chan_id] = execution_end_time
                execution_total_elapsed_time += execution_end_time

                execution_start_time = 0
                execution_end_time = 0


        if self.config.execution == "train":
            columns = list(losses_df.columns)
            columns_number=1
            for elem in columns:
                if "train" in elem:
                    columns_number+=1

            losses_df["train_loss_"+str(columns_number)] = train_loss
            losses_df["valid_loss_"+str(columns_number)] = validation_loss

            losses_df.to_csv('./data/{}/losses.csv'.format(self.id), sep=',', index=False)



        if self.config.execution != "train" and self.config.execution != "find_hp":
            self.log_final_stats()

        if self.config.execution == "find_hp":
            first_part_path = "./hp/hp_times".format(self.id)
            training_times_row["total_elapsed_time"] = training_total_elapsed_time
            write_csv(first_part_path, training_times_row, col_header)

        if self.config.execution == "train_and_predict":

            training_times_row["total_elapsed_time"] = training_total_elapsed_time
            execution_times_row["total_elapsed_time"] = execution_total_elapsed_time

            write_csv("./results/training_times", training_times_row, col_header)
            write_csv("./results/execution_times", execution_times_row, col_header)


    def run(self):
        """
        Initiate processing for all channels.
        """
        logger.info("Execution mode: {}\n".format(self.config.execution))

        if self.config.execution == "search_p":
            # header for csv file
            col_header = ["P", "Precision", "Recall", "Total True Positives", "Total False Positives", "Total False Negatives",
                          "Total True Negatives"]

            df = pd.DataFrame(columns=col_header)

            # creates the array of values for p
            # for p values tending to 1 recall tends to 0 and precision tends to 1
            # then we limit the p values from 0.01 to 0.35
            array = np.linspace(0.01, 0.35, 35).tolist()
            formatted_array = [round(elem, 2) for elem in array]

            i=0
            for elem in formatted_array:
                self.config.p = elem

                self.execute_detection()

                row = [self.config.p, self.precision, self.recall, self.result_tracker['true_positives'],
                       self.result_tracker['false_positives'], self.result_tracker['false_negatives'],
                       self.result_tracker['true_negatives']]

                df.loc[i] = row
                i += 1

                self.result_tracker = {
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_negatives': 0
                }
            df.to_csv('./data/{}/p_values.csv'.format(self.id), sep=',')

        else:
            self.execute_detection()


