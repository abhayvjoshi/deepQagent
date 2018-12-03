import tensorflow as tf
import random
import numpy as np
from pacman import Directions
from collections import deque
import sys
import time
import game

inputs = {
    'save_interval': 10000,
    'mem_size': 100000,
    'save_file': None,
    'load_file': None,
    'train_start': 50,
    'batch_size': 32,
    'discount': 0.95,
    'eps_final': 0.1,
    'lr': .0002,
    'eps': 1.0,
    'eps_step': 1
}


class deepQAgents(game.Agent):
    def __init__(self, args):
        # print "DQN Agent is initializing"
        self.parameter = inputs
        self.parameter['height'], self.parameter['width'], self.parameter['num_training'] = args['height'], args['width'], args['numTraining']
        opt_graphic = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.time_period = tf.Session(config=tf.ConfigProto(gpu_options=opt_graphic))
        self.Network_Q = DQN(self.parameter)
        self.gr_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.world_Q, self.s = [], time.time()
        self.count = self.Network_Q.sess.run(self.Network_Q.global_step)
        self.display_amount, self.class_count,self.final_result,self.nps, self.final_bonus = 0, 0, 0, 0, 0.
        self.final_results, self.memory_play_again = deque(), deque()


    def combineStateMatrices(self, st_M):
        st_M = np.swapaxes(st_M, 0, 2)
        all_val = np.zeros((7, 7))
        for i in range(len(st_M)):
            all_val += (i + 1) * st_M[i] / 6
        return all_val

    def exploit(self):
        self.prediction_Q = self.Network_Q.sess.run(self.Network_Q.y, feed_dict = {self.Network_Q.x: np.reshape(self.present_st,
                                (1, self.parameter['width'], self.parameter['height'], 6)),
                               self.Network_Q.q_t: np.zeros(1),self.Network_Q.actions: np.zeros((1, 4)),self.Network_Q.terminals: np.zeros(1),
                               self.Network_Q.rewards: np.zeros(1)})[0]
        temp = max(self.prediction_Q)
        self.world_Q.append(temp)
        won_stat = np.argwhere(self.prediction_Q == np.amax(self.prediction_Q))
        return self.obtain_dir(won_stat[np.random.randint(0, len(won_stat))][0]) if len(won_stat) > 1 else self.obtain_dir(won_stat[0][0])

    def final(self, state):
        self.terminal = True
        self.ep_rew += self.final_bonus
        self.observation_step(state)
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.nps, self.class_count, self.count, time.time() - self.s, self.ep_rew, self.parameter['eps']))
        sys.stdout.flush()

    def getAction(self, s):
        move = self.obtainAction(s)
        if move not in s.getLegalActions(0):
            move = Directions.STOP
        return move

    def getStateMatrices(self, s):
        def obtainFoodM(s):
            w, h = s.data.layout.width, s.data.layout.height
            grid, mat = s.data.food, np.zeros((h, w), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    cell = 1 if grid[j][i] else 0
                    mat[-1-i][j] = cell
            return mat

        def obtainPacM(s):
            w, h = s.data.layout.width, s.data.layout.height
            mat = np.zeros((h, w), dtype=np.int8)
            for agentState in s.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    mat[-1-int(pos[1])][int(pos[0])] = cell
            return mat

        def obtainWallM(s):
            width, height = s.data.layout.width, s.data.layout.height
            grid, matrix = s.data.layout.walls, np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def obtainScaredGoM(state):
            w, h = state.data.layout.width, state.data.layout.height
            mat = np.zeros((h, w), dtype=np.int8)
            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        mat[-1-int(pos[1])][int(pos[0])] = cell
            return mat

        def obtainCapM(s):
            w, h = s.data.layout.width, s.data.layout.height
            capsules, mat = s.data.layout.capsules, np.zeros((h, w), dtype=np.int8)
            for i in capsules:
                mat[-1-i[1], i[0]] = 1
            return mat

        def obtainGhostM(s):
            w, h = s.data.layout.width, s.data.layout.height
            mat = np.zeros((h, w), dtype=np.int8)
            for agentState in s.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        mat[-1-int(pos[1])][int(pos[0])] = cell
            return mat

        w, h = self.parameter['width'], self.parameter['height']
        obMat = np.zeros((6, h, w))
        obMat[0],obMat[1],obMat[2],obMat[3],obMat[4],obMat[5]  = obtainWallM(s),obtainPacM(s),obtainGhostM(s),obtainScaredGoM(s),obtainFoodM(s),obtainCapM(s)
        obMat = np.swapaxes(obMat, 0, 2)
        return obMat

    def obtainAction(self, s):
        if np.random.rand() <= self.parameter['eps']:
            push = self.obtain_dir(np.random.randint(0, 4))
        else:
            push = self.exploit()
        self.act_final = self.obtain_v(push)
        return push

    def observation_step(self, s):
        if self.act_final is not None:
            self.final_st, self.present_st  = np.copy(self.present_st), self.getStateMatrices(s)
            self.present_record,self.final_result  = s.getScore(), self.present_record
            bonus = self.present_record - self.final_result
            if bonus < -10:
                self.won = False
            self.final_bonus = 50. if bonus > 20 else 10. if bonus > 0 else -1. if bonus < 0 and bonus >= -10 else -500.
            if(self.terminal and self.won):
                self.final_bonus = 100.
            self.ep_rew += self.final_bonus
            evnts = (self.final_st, float(self.final_bonus), self.act_final, self.present_st, self.terminal)
            self.memory_play_again.append(evnts)
            if len(self.memory_play_again) > self.parameter['mem_size']:
                self.memory_play_again.popleft()
            if(inputs['save_file']):
                if self.class_count > self.parameter['train_start'] and self.class_count % self.parameter['save_interval'] == 0:
                    self.Network_Q.save_ckpt('saves/model-' + inputs['save_file'] + "_" + str(self.count) + '_' + str(self.nps))
                    print('Model saved')
            self.train()
        self.frame += 1
        self.class_count += 1
        self.parameter['eps'] = max(self.parameter['eps_final'], 1.00 - float(self.count) / float(self.parameter['eps_step']))

    def observationFunction(self, state):
        self.observation_step(state)
        self.terminal = False

        return state

    def obtain_1hot(self, a):
        a_hot = np.zeros((self.parameter['batch_size'], 4))
        for i in range(len(a)):
            a_hot[i][int(a[i])] = 1
        return a_hot

    def obtain_dir(self, value):
        return Directions.NORTH if value == 0. else Directions.EAST if value == 1. else Directions.SOUTH if value == 2. else Directions.WEST

    def obtain_v(self, direction):
        return 0 if dir == Directions.NORTH else 1 if dir == Directions.EAST else 2 if dir == Directions.SOUTH else 3

    def registerInitialState(self, s):
        self.final_result, self.present_record, self.final_bonus, self.ep_rew = 0, 0, 0., 0
        self.final_st, self.present_st = None, self.getStateMatrices(s)
        self.act_final, self.terminal, self.won,self.world_Q = None, None, True, []
        self.delay, self.frame = 0, 0
        self.nps = self.nps + 1

    def train(self):
        if (self.class_count > self.parameter['train_start']):
            btch = random.sample(self.memory_play_again, self.parameter['batch_size'])
            s_b, r_b, a_b, n_b, t_b = [], [], [], [], []
            for i in btch:
                t_b.append(i[4])
                n_b.append(i[3])
                a_b.append(i[2])
                r_b.append(i[1])
                s_b.append(i[0])
            s_b,r_b, a_b, n_b, t_b = np.array(s_b), np.array(r_b), self.obtain_1hot(np.array(a_b)), np.array(n_b), np.array(t_b)
            self.count, self.display_amount = self.Network_Q.train(s_b, a_b, t_b, n_b, r_b)


class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x')
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')

        # Layer 1 (Convolutional)
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')

        # Layer 2 (Convolutional)
        layer_name = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1
        self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')

        o2_shape = self.o2.get_shape().as_list()

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips')
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        # Layer 4
        layer_name = 'fc4' ; hiddens = 4 ; dim = 256
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs')

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # self.optim = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)
        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.params['load_file'])


    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)
