from telemanom.ESN import SimpleESN
from telemanom.ESNnoserializzazione import ESNnoser
import yaml
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.tuners import RandomSearch

import tensorflow as tf

from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import random

SEED = 42

class MyHyperModel(HyperModel):
    def __init__(self, config, channel, model, layers):
        self.config = config
        self.channel = channel
        self.model = model
        self.layers = layers


    def build(self, hp):
        if self.model == "ESN":
            units = hp.Choice("units",[100, 200, 300, 400, 500, 600, 700, 800, 900, 950])
            model = ESNnoser(config=self.config,
                              units=units,
                              input_scaling=hp.Choice("input_scaling",[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                              spectral_radius=hp.Choice("spectral_radius",
                                                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]),
                              leaky=hp.Float("leaky", 0.1, 1, 0.10),
                              layers=self.layers,
                              circular_law=True,
                              SEED=SEED
                              )
            model.build(input_shape=(self.channel.X_train.shape[0], self.channel.X_train.shape[1], self.channel.X_train.shape[2]))


        elif self.model == "LSTM":
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            if self.layers > 1:
                units = hp.Choice("units", values=[1,5,10,15,20,25])
            else:
                units = hp.Choice("units", values=[1, 5, 10, 15, 20, 25, 30, 35, 38])

            model = Sequential()

            for i in range(self.layers - 1):
                model.add(LSTM(units=units,
                               input_shape=(None, self.channel.X_train.shape[2]),
                               return_sequences=True))
                model.add(Dropout(hp.Float("dropout", 0.1, 0.5, 0.05)))

            model.add(LSTM(units=units,
                           return_sequences=False))
            model.add(Dropout(hp.Float("dropout", 0.1, 0.5, 0.05)))

            model.add(Dense(
                self.config.n_predictions))


        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss=self.config.loss_metric,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate))

        return model

class FindHP():
    def __init__(self, id, channel, config, model):
        self.id = id
        self.channel = channel
        self.config = config
        self.model = model

        self.run()

    def run(self):
        layers = 2
        tuner = RandomSearch(
            MyHyperModel(config=self.config, channel=self.channel, model=self.model, layers=layers),
            objective="val_loss",
            directory='hp/{}/kerastuner'.format(self.id),
            max_trials=50,
            project_name=self.channel.id,
        )

        # default hp
        if self.model == "ESN":
            hp = {
                'units': 100,
                'input_scaling': 1,
                'radius': 0.99,
                'leaky': 1,
                'layers': layers
            }
            tuner.search(self.channel.X_train,
                         self.channel.y_train,
                         epochs=5,
                         validation_data=(self.channel.X_valid, self.channel.y_valid),
                         verbose=1
                         )
        elif self.model == "LSTM":
            hp = {
                'units': 1,
                'dropout': 0.10,
                'layers': layers,
            }
            batch_size = self.config.lstm_batch_size
            tuner.search(self.channel.X_train,
                         self.channel.y_train,
                         batch_size=batch_size,
                         epochs=5,
                         validation_data=(self.channel.X_valid, self.channel.y_valid),
                         verbose=1
                         )


        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        if self.model == "ESN":
            hp["units"] = best_hps.get('units')
            hp["input_scaling"] = float("{:.2f}".format(best_hps.get('input_scaling')))
            hp["radius"] = float("{:.2f}".format(best_hps.get('spectral_radius')))
            hp["leaky"] = float("{:.2f}".format(best_hps.get('leaky')))

        if self.model == "LSTM":
            hp["units"] = best_hps.get('units')
            hp["dropout"] = float("{:.2f}".format(best_hps.get('dropout')))

        hp["learning_rate"] = float(format(best_hps.get('learning_rate')))

        f = open(f'./hp/{self.id}/config/{self.channel.id}.yaml', "w")

        yaml.dump(hp, f, default_flow_style=False)
        f.close()