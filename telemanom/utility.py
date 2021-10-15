from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telemanom.ESN import SimpleESN


def create_lstm_model(channel,config, hp):
    model = Sequential()
    if len(hp) == 0:
        model.add(LSTM(
            config.layers[0],
            input_shape=(None, channel.X_train.shape[2]),
            return_sequences=True))
        model.add(Dropout(config.dropout))
        model.add(LSTM(
            config.layers[1],
            return_sequences=False))
        model.add(Dropout(config.dropout))
        model.add(Dense(
            config.n_predictions))

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

    else:
        units = int(hp["units"])
        dropout = float(hp["dropout"])
        learning_rate = float(hp["learning_rate"])
        layers = int(hp["layers"])

        for i in range(layers-1):
            model.add(LSTM(
                units,
                input_shape=(None, channel.X_train.shape[2]),
                return_sequences=True))
            model.add(Dropout(dropout))

        model.add(LSTM(
            units,
            input_shape=(None, channel.X_train.shape[2]),
            return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense(
            config.n_predictions))

        model.compile(loss=config.loss_metric,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model

def create_esn_model(channel,config, hp, seed):
    if len(hp) == 0:
        model = SimpleESN(config=config,
                              SEED=seed,
                              circular_law=config.circular_law,
                             )
        model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))
        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

    else:
        model = SimpleESN(config=config,
                          units=int(hp["units"]),
                          input_scaling=float(hp["input_scaling"]),
                          spectral_radius=float(hp["radius"]),
                          leaky=float(hp["leaky"]),
                          SEED=seed,
                          layers=int(hp["layers"]),
                          circular_law=config.circular_law
                          )

        model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))
        model.compile(loss=config.loss_metric,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=float(hp["learning_rate"])))

    return model



