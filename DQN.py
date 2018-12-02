import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Input, Flatten, Embedding,Activation
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from tqdm import tqdm


class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'
        self.model = self.build_model()
        self.count =0

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=(7, 7, 6, ),activation='relu'))
        # model.add(Conv2D(32, (3,3), padding='same', input_shape=(7,  7, 6)))
        # model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=0.0091))
        plot_model(model, to_file='mdl.png', show_shapes=True)
        return model

    def train(self, bat_s, bat_a, bat_t, bat_n, bat_r):
        self.count+=1
        curr_st = bat_s
        next_st = bat_n
        target = bat_r
        for i in xrange(len(bat_t)):
            if not bat_t[i]:
                target[i] = (bat_r[i] + 0.95 *
                          np.amax(self.model.predict(next_st)))
        target_f = self.model.predict(curr_st)
        for i in xrange(len(bat_t)):
            target_f[i][int(bat_a[i])] = target[i]
        # print target_f
        # print bat_a[i]
        self.model.fit(curr_st, target_f, epochs=6, verbose=0)
        return self.model,self.count