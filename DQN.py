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

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        #model.add(Dense(24, input_shape=(7, 7, 6, ),activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', input_shape = (7,  7, 6)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=0.091))
        plot_model(model, to_file='mdl.png', show_shapes=True)
        return model

    def train(self, bat_s, bat_a, bat_t, bat_n, bat_r):
        for i in tqdm(xrange(32)):
            curr_st = bat_s[i].reshape(1, 7, 7, 6)
            next_st = bat_n[i].reshape(1, 7, 7, 6)
            target = bat_r[i]
            if not bat_t[i]:
                target = (bat_r[i] + 0.95 *
                          np.amax(self.model.predict(next_st)))
            target_f = self.model.predict(curr_st)
            target_f[0][int(bat_a[i])] = target
            self.model.fit(curr_st, target_f, epochs=6, verbose=0)
        return self.model