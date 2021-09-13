import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
import os
import random
import logging

def sparse_eye(M):
    # Generates an M x M matrix to be used as sparse identity matrix for the
    # re-scaling of the sparse recurrent kernel in presence of non-zero leakage.
    # The neurons are connected according to a ring topology, where each neuron
    # receives input only from one neuron and propagates its activation only to one other neuron.
    # All the non-zero elements are set to 1
    dense_shape = (M, M)

    # gives the shape of a ring matrix:
    indices = np.zeros((M, 2))
    for i in range(M):
        indices[i, :] = [i, i]
    values = np.ones(shape=(M,)).astype('f')

    W = (tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)))
    return W

def sparse_recurrent_tensor(M, C=1):
    # Generates an M x M matrix to be used as sparse recurrent kernel
    # For each column only C elements are non-zero
    # (i.e., each recurrent neuron takes input from C other recurrent neurons).
    # The non-zero elements are generated randomly from a uniform distribution in [-1,1]

    dense_shape = (M, M)  # the shape of the dense version of the matrix

    indices = np.zeros((M * C, 2))  # indices of non-zero elements initialization
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, :] = [idx[j], i]
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    W = (tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)))
    return W

class ReservoirCell(keras.layers.Layer):
    # Implementation of a shallow reservoir to be used as cell of a Recurrent Neural Network
    # The implementation is parametrized by:
    # units - the number of recurrent neurons in the reservoir
    # input_scaling - the max abs value of a weight in the input-reservoir connections
    #                 note that whis value also scales the unitary input bias
    # spectral_radius - the max abs eigenvalue of the recurrent weight matrix
    # leaky - the leaking rate constant of the reservoir
    # connectivity_input - number of outgoing connections from each input unit to the reservoir

    def __init__(self, units, SEED,
                 input_scaling=1., spectral_radius=0.99, leaky=1,connectivity_recurrent=1,
                 circular_law=False,
                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_recurrent = connectivity_recurrent
        self.SEED = SEED
        self.circular_law = circular_law,
        super().__init__(**kwargs)

    def build(self, input_shape):
        # build the input weight matrix
        self.kernel = tf.random.uniform((input_shape[-1], self.units), minval=-1, maxval=1) * self.input_scaling

        # build the recurrent weight matrix
        if self.circular_law:
            # uses circular law to determine the values of the recurrent weight matrix
            value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))
            self.recurrent_kernel = tf.random.uniform(shape=(self.units, self.units), minval=-value, maxval=value)

        else:
            W = sparse_recurrent_tensor(self.units, C=self.connectivity_recurrent)

            # re-scale the weight matrix to control the effective spectral radius of the linearized system
            if (self.leaky == 1):
                # if no leakage then rescale the W matrix
                # compute the spectral radius of the randomly initialized matrix
                e, _ = tf.linalg.eig(tf.sparse.to_dense(W))
                rho = max(abs(e))
                # rescale the matrix to the desired spectral radius
                W = W * (self.spectral_radius / rho)
                self.recurrent_kernel = W
            else:
                I = sparse_eye(self.units)
                W2 = tf.sparse.add(I * (1 - self.leaky), W * self.leaky)
                e, _ = tf.linalg.eig(tf.sparse.to_dense(W2))
                rho = max(abs(e))
                W2 = W2 * (self.spectral_radius / rho)
                self.recurrent_kernel = tf.sparse.add(W2, I * (self.leaky - 1)) * (1 / self.leaky)

        self.bias = tf.random.uniform(shape=(self.units,), minval=-1, maxval=1) * self.input_scaling

        self.built = True

    def call(self, inputs, states):
        # computes the output of the cell given the input and previous state
        prev_output = states[0]
        input_part = tf.matmul(inputs, self.kernel)
        if self.circular_law:
            state_part = tf.matmul(prev_output, self.recurrent_kernel)
        else:
            state_part = tf.sparse.sparse_dense_matmul(prev_output, self.recurrent_kernel)

        output = prev_output * (1 - self.leaky) + tf.nn.tanh(input_part + self.bias + state_part) * self.leaky

        return output, [output]

    def get_config(self):
        base_config = super().get_config()

        return {**base_config,
                "units": self.units,
                "spectral_radius": self.spectral_radius,
                "leaky": self.leaky,
                "input_scaling": self.input_scaling,
                "state_size": self.state_size
                }

    def from_config(cls, config):
        return cls(**config)

class ESNnoser(keras.Model):
    def __init__(self, units=100, input_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 config=None, SEED=42, layers=1, connectivity_recurrent=10,
                 circular_law=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.units = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.n_layers = layers
        self.connectivity_recurrent = connectivity_recurrent
        self.circular_law = circular_law

        self.SEED = SEED

        if self.SEED == 42:
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        os.environ['PYTHONHASHSEED'] = str(self.SEED)

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)

        self.reservoir = Sequential()
        for i in range(self.n_layers - 1):
            self.reservoir.add(tf.keras.layers.RNN(cell=ReservoirCell(
                units=units,
                spectral_radius=spectral_radius, leaky=leaky,
                connectivity_recurrent=connectivity_recurrent,
                input_scaling=input_scaling, circular_law=self.circular_law,
                SEED=self.SEED),
                return_sequences=True
            )
            )
        # last reservoir
        self.reservoir.add(tf.keras.layers.RNN(cell=ReservoirCell(
            units=units,
            spectral_radius=spectral_radius, leaky=leaky,
            connectivity_recurrent=connectivity_recurrent,
            input_scaling=input_scaling, circular_law=self.circular_law,
            SEED=self.SEED)
        )
        )

        self.readout = Sequential()
        self.readout.add(tf.keras.layers.Dense(config.n_predictions))
        self.readout.compile(loss=config.loss_metric, optimizer=config.optimizer)

    def call(self, inputs):
        r = self.reservoir(inputs)
        y = self.readout(r)
        return y

    def get_config(self):
        return {"units": self.units,
                "input_scaling": self.input_scaling,
                "spectral_radius": self.spectral_radius,
                "leaky": self.leaky,
                "config": self.config,
                "connectivity_recurrent":self.connectivity_recurrent,
                "n_layers": self.n_layers}

    def from_config(cls, config):
        return cls(**config)

    def generator(self, dataset):
        ds = dataset.repeat().prefetch(tf.data.AUTOTUNE)
        iterator = iter(ds)
        x, y = iterator.get_next()

        while True:
            yield x, y

    #da eliminare
    def secondsToStr(self, t):
        from functools import reduce
        return "%dh:%02dm:%02ds.%03dms" % \
               reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                      [(t * 1000,), 1000, 60, 60])

    def fit(self, x, y, **kwargs):
        #import time
        #logger = logging.getLogger('tests.log')

        #start_time = time.time()

        X_train = self.reservoir(x)
        x_val, y_val = kwargs['validation_data']
        x_val_1 = self.reservoir(x_val)
        kwargs['validation_data'] = (x_val_1, y_val)




        return self.readout.fit(X_train,y, **kwargs)