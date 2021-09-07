from telemanom.ESNnoserializzazione import ESNnoser
import pandas as pd
from telemanom.channel import Channel
from telemanom.helpers import Config

from tensorflow.keras.callbacks import History, EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.autograph.set_verbosity(0)

epochs = 5
units = 800
input_scaling=1
spectral_radius=0.99
leaky=0.8
layers=1

config_path = "config.yaml"
config = Config(config_path)

chan_df = pd.read_csv("labeled_anomalies.csv")

cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=config.patience,
                                        min_delta=config.min_delta,
                                        verbose=0)]
columns = ["model", "init time", "build time", "write/read time", "fit time"]
df = pd.DataFrame(columns=columns)

for i, row in chan_df.iterrows():
    chan_id = row.chan_id

    channel = Channel(config, row.chan_id)
    channel.load_data()

    print("Channel name: "+chan_id)
    print("Number of sequences: "+str(len(channel.X_train)))

    model = ESNnoser(config=config,
                     units=units,
                     input_scaling=input_scaling,
                     spectral_radius=spectral_radius,
                     leaky=leaky,
                     layers=layers
                     )

    model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))

    model.compile(loss=config.loss_metric, optimizer=config.optimizer)

    model.fit(channel.X_train,
              channel.y_train,
              validation_data=(channel.X_valid, channel.y_valid),
              epochs=epochs,
              callbacks=cbs,
              batch_size=32,
              verbose=0)