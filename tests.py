from telemanom.ESN import SimpleESN
from telemanom.ESNnoprecalcolo import ESNnoprecalcolo
from telemanom.ESNnoserializzazione import ESNnoser
import pandas as pd
from telemanom.channel import Channel
from telemanom.helpers import Config
import time
from tensorflow.keras.callbacks import History, EarlyStopping
from functools import reduce
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.autograph.set_verbosity(0)

#logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(filename="tests.log", level=logging.INFO)

epochs = 150
units=1000
input_scaling=1
spectral_radius=0.99
leaky=0.8
layers=1


def esn_precalcolo_serializzazione(config):
    logging.info("Model: ESN precalcolo+serializzazione")
    start_time = time.time()
    model = SimpleESN(config=config,
                      units=units,
                      input_scaling=input_scaling,
                      spectral_radius=spectral_radius,
                      leaky=leaky,
                      layers=layers
                      )
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo inizializzazione: " + time_string)

    start_time = time.time()

    model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo build: " + time_string)

    start_time = time.time()

    model.compile(loss=config.loss_metric, optimizer=config.optimizer)
    end_time = time.time() - start_time

    time_string = secondsToStr(end_time)
    logging.info("Tempo compile: " + time_string)

    start_time = time.time()

    model.fit(channel.X_train,
              channel.y_train,
              validation_data=(channel.X_valid, channel.y_valid),
              epochs=150,
              callbacks=cbs,
              verbose=True)
    end_time = time.time() - start_time

    time_string = secondsToStr(end_time)
    logging.info("Tempo precalcolo+scrittura+rilettura+allenamento: " + time_string + "\n")

def esn_noprecalcolo(config):
    logging.info("Model: ESN no precalcolo")
    start_time = time.time()
    model = ESNnoprecalcolo(config= config,
                      units=units,
                      input_scaling=input_scaling,
                      spectral_radius=spectral_radius,
                      leaky=leaky,
                      layers=layers
                      )
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo inizializzazione: "+time_string)

    start_time = time.time()
    model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo build: " + time_string)

    start_time = time.time()
    model.compile(loss=config.loss_metric, optimizer=config.optimizer)
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo compile: " + time_string)

    start_time = time.time()
    model.fit(channel.X_train,
              channel.y_train,
              validation_data=(channel.X_valid, channel.y_valid),
              epochs=epochs,
              callbacks=cbs,
              batch_size=32,
              verbose=True)
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo allenamento: " + time_string+"\n")


def esn_noserializzazione(config):
    logging.info("Model: ESN noserializzazione")
    start_time = time.time()
    model = ESNnoser(config=config,
                      units=units,
                      input_scaling=input_scaling,
                      spectral_radius=spectral_radius,
                      leaky=leaky,
                      layers=layers
                      )
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo inizializzazione: " + time_string)

    start_time = time.time()
    model.build(input_shape=(channel.X_train.shape[0], channel.X_train.shape[1], channel.X_train.shape[2]))
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo build: " + time_string)

    start_time = time.time()
    model.compile(loss=config.loss_metric, optimizer=config.optimizer)
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo compile: " + time_string)

    start_time = time.time()
    model.fit(channel.X_train,
              channel.y_train,
              validation_data=(channel.X_valid, channel.y_valid),
              epochs=epochs,
              callbacks=cbs,
              batch_size=32,
              verbose=True)
    end_time = time.time() - start_time
    time_string = secondsToStr(end_time)
    logging.info("Tempo precalcolo+allenamento: " + time_string + "\n")


def secondsToStr(t):
    return "%dh:%02dm:%02ds.%03dms" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])



import tensorflow as tf

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

    if chan_id != "P-10":
        continue

    channel = Channel(config, row.chan_id)
    channel.load_data()

    logging.info("Channel name: "+chan_id)
    logging.info("Number of sequences: "+str(len(channel.X_train)))

    esn_precalcolo_serializzazione(config)

    #no precalcolo no serializzazione
    esn_noprecalcolo(config)

    #si precalcolo no serializzazione
    esn_noserializzazione(config)




