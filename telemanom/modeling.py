import yaml
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, EarlyStopping
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from telemanom.utility import create_lstm_model, create_esn_model
import random

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')

def get_seed(folder, chan_id):
    path = f'./data/{folder}/models/seeds.log'
    file1 = open(path, 'r')
    seed = 0
    while True:
        line = file1.readline()
        # if line is empty end of file is reached
        if not line:
            break

        strings = line.strip().split(" ")
        channel_id = strings[0]
        seed = int(strings[1])

        if channel_id == chan_id:
            break

    file1.close()
    return seed


class Model:
    def __init__(self, config, run_id, channel):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None

        #da sistemare pure questa
        #if not self.config.train and not self.config.train_only:
        if self.config.execution == "predict":
            try:
                logger.info('Loading pre-trained model')
                hp = {}
                if self.config.load_hp:
                    logger.info('Loading hp id: {}'.format(self.config.hp_research_id))
                    path = "./hp/{}/config/{}.yaml".format(self.config.hp_research_id, self.chan_id)
                    with open(path, 'r') as file:
                        hp = yaml.load(file, Loader=yaml.BaseLoader)

                if self.config.model_architecture != "LSTM":
                    # get seed for that model
                    seed = get_seed(self.config.use_id, self.chan_id)
                    self.model = create_esn_model(channel, self.config, hp, seed)

                    self.model.load_weights(os.path.join('data', self.config.use_id,
                                                         'models', self.chan_id + '.h5'))


                else:
                    self.model = load_model(os.path.join('data', self.config.use_id,
                                                         'models', self.chan_id + '.h5'))
            except (FileNotFoundError, OSError) as e:
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))

                self.train_new(channel)
                self.save()

        elif self.config.execution == "train" or self.config.execution == "train_and_predict":
            self.train_new(channel)
            self.save()

        else:
            logger.info("Configuration file error, check execution flag")
            sys.exit("Configuration file error, check execution flag")


    def train_new(self, channel):
        """
        Train ESN or LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        hp = {}
        if self.config.load_hp:
            path = os.path.join("hp", self.config.hp_research_id, "config", "{}.yaml".format(self.chan_id))
            try:
                with open(path, 'r') as file:
                    hp = yaml.load(file, Loader=yaml.BaseLoader)
                    print(hp)

                if self.config.model_architecture == "ESN":
                    logger.info('units: {}'.format(hp["units"]))
                    logger.info('input_scaling: {}'.format(hp["input_scaling"]))
                    logger.info('radius: {}'.format(hp["radius"]))
                    logger.info('leaky: {}'.format(hp["leaky"]))
                    logger.info('learning_rate: {}'.format(hp["learning_rate"]))

                if self.config.model_architecture == "LSTM":
                    logger.info('units: {}'.format(hp["units"]))
                    logger.info('dropout: {}'.format(hp["dropout"]))
                    logger.info('learning_rate: {}'.format(hp["learning_rate"]))
                    logger.info('layers: {}'.format(hp["layers"]))

            except FileNotFoundError as e:
                logger.info("No configuration file at {} using default hypeparameters".format(path))
                raise e
        else:
            logger.info("default hp")


        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        if self.config.model_architecture == "LSTM":

            self.model = create_lstm_model(channel,self.config, hp)


            self.history = self.model.fit(channel.X_train,
                                          channel.y_train,
                                          batch_size=self.config.lstm_batch_size,
                                          epochs=self.config.epochs,
                                          validation_data=(channel.X_valid, channel.y_valid),
                                          callbacks=cbs,
                                          verbose=True)

        #esn
        else:
            SEED = random.randint(43, 999999999)
            if self.config.model_architecture == "ESN":
                self.model = create_esn_model(channel,self.config, hp, SEED)
                self.history = self.model.fit(channel.X_train,
                                              channel.y_train,
                                              validation_data=(channel.X_valid, channel.y_valid),
                                              epochs=self.config.epochs,
                                              callbacks=cbs,
                                              verbose=True)



        logger.info('validation_loss: {}\n'.format(self.history.history["val_loss"][-1]))

    def save(self):
        """
        Save trained model, loss and validation loss graphs .
        """

        if self.config.save_graphs:
            plt.figure()
            plt.plot(self.history.history["loss"], label="Training Loss")
            plt.plot(self.history.history["val_loss"], label="Validation Loss")
            plt.title(f'Training and validation loss model: {self.config.model_architecture} channel: {self.chan_id}')

            plt.legend()

            plt.savefig(os.path.join('data', self.run_id, 'images',
                                     '{}_loss.png'.format(self.chan_id)))
            #plt.show()
            plt.close()

        if self.config.model_architecture != "LSTM":
            self.model.save_weights(os.path.join('data', self.run_id, 'models',
                                         '{}.h5'.format(self.chan_id)))
            #saving seeds
            path = './data/{}/models/seeds.log'.format(self.run_id)
            f = open(path, "a")
            f.write("{} {}\n".format(self.chan_id, self.model.SEED))
            f.close()


        else:
            self.model.save(os.path.join('data', self.run_id, 'models',
                                         '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='mean'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM or ESN model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))


        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model.predict(X_test_batch)
            self.aggregate_predictions(y_hat_batch)


        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat

        np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'
                             .format(self.chan_id)), self.y_hat)

        return channel
