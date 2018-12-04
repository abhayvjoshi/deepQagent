import tensorflow as tf
import random
import numpy as np
from pacman import Directions
from collections import deque
import sys
import time
import game
import json



class deepQAgents(game.Agent):
    def __init__(self, args):
        with open('default_config.json') as f:
            self.parameter = json.load(f)
        self.parameter['width'] = args['width']
        self.parameter['height'] = args['height']
        self.parameter['num_training'] = args['numTraining']
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.nn = DQN(self.parameter)
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.World_Q = []
        self.val_d = 0
        self.count = self.nn.sess.run(self.nn.global_step)
        self.class_count = 0
        self.num_of_episodes = 0
        self.prev_result = 0
        self.s = time.time()
        self.prev_bonus = 0.
        self.memory_rep = deque()
        self.prev_results = deque()


    def final(self, state):
        self.ep_rew += self.prev_bonus
        self.terminal = True
        self.observation_step(state)
        print("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f \n" %
                         (self.num_of_episodes, self.class_count, self.count, time.time() - self.s, self.ep_rew, self.parameter['eps']))
        # sys.stdout.flush()

    def getAction(self, state):
        move = self.getMove(state)
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move

    def get_direction(self, val):
        # if value == 0:
        #     return Directions.NORTH
        # elif value == 1:
        #     return Directions.SOUTH
        # elif value == 2:
        #     return Directions.EAST
        # else:
        #     return Directions.WEST
        return Directions.NORTH if val == 0. else Directions.EAST if val == 1. else Directions.SOUTH if val == 2. else Directions.WEST


    def getMove(self, s):
        if np.random.rand() > self.parameter['eps']:
            self.Q_pred = self.nn.sess.run(
                self.nn.y,
                feed_dict = {self.nn.x: np.reshape(self.present_st,
                                                   (1, self.parameter['width'], self.parameter['height'], 6)),
                             self.nn.q_t: np.zeros(1),
                             self.nn.actions: np.zeros((1, 4)),
                             self.nn.terminals: np.zeros(1),
                             self.nn.rewards: np.zeros(1)})[0]

            self.World_Q.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            move = self.get_direction(np.random.randint(0, 4))
        self.prev_act = self.get_value(move)
        return move


    def getStateMatrices(self, s):
        width, height = self.parameter['width'], self.parameter['height']
        input_matrix = np.zeros((6, width, height))
        pacman_matrix = np.zeros((width, height))
        ghost_matrix = np.zeros((width, height))
        scared_matrix = np.zeros((width, height))
        capsule_matrix = np.zeros((width, height))
        input_matrix[0] = np.array(s.data.layout.walls.data, int)
        for i in s.data.agentStates:
            if i.isPacman:
                pacman_pos = i.getPosition()
                pacman_matrix[int(pacman_pos[0])][int(pacman_pos[1])] = 1
            else:
                if not i.scaredTimer > 0:
                    ghost_pos = i.getPosition()
                    ghost_matrix[int(ghost_pos[0])][int(ghost_pos[1])] = 1
                else:
                    scared_ghost_pos = i.getPosition()
                    scared_matrix[int(scared_ghost_pos[0])][int(scared_ghost_pos[1])] = 1
        input_matrix[1] = pacman_matrix
        input_matrix[2] = ghost_matrix
        input_matrix[3] = scared_matrix
        input_matrix[4] = np.array(s.data.food.data, int)
        for i in s.data.capsules:
            capsule_matrix[i[0]][i[1]] = 1
        input_matrix[5] = capsule_matrix
        return np.swapaxes(input_matrix, 0, 2)

    def get_value(self, dir):
        # if direction == Directions.NORTH:
        #     return 0
        # elif direction == Directions.SOUTH:
        #     return 1
        # elif direction == Directions.EAST:
        #     return 2
        # else:
        #     return 3
        return 0 if dir == Directions.NORTH else 1 if dir == Directions.EAST else 2 if dir == Directions.SOUTH else 3

    def mergeStateMatrices(self, stateMatrices):
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total


    def observation_step(self, s):
        if self.prev_act is not None:
            self.prev_st = np.copy(self.present_st)
            self.present_st = self.getStateMatrices(s)
            self.present_result = s.getScore()
            bonus = self.present_result - self.prev_result
            self.prev_result = self.present_result
            if bonus > 20:
                self.prev_bonus = 50.
            elif bonus > 0:
                self.prev_bonus = 10.
            elif bonus < -10:
                self.prev_bonus = -500.
                self.won = False
            elif bonus < 0:
                self.prev_bonus = -1.
            if(self.terminal and self.won):
                self.prev_bonus = 100.
            self.ep_rew += self.prev_bonus
            exp = (self.prev_st, float(self.prev_bonus), self.prev_act, self.present_st, self.terminal)
            self.memory_rep.append(exp)
            if len(self.memory_rep) > self.parameter['mem_size']:
                self.memory_rep.popleft()
            self.train()
        self.class_count += 1
        self.frame += 1
        self.parameter['eps'] = max(self.parameter['eps_final'],
                                    1.00 - float(self.count) / float(self.parameter['eps_step']))

    def observationFunction(self, s):
        self.terminal = False
        self.observation_step(s)
        return s

    def obtain_1hot(self, act):
        actions_onehot = np.zeros((self.parameter['batch_size'], 4))
        for i in range(len(act)):
            actions_onehot[i][int(act[i])] = 1
        return actions_onehot

    def registerInitialState(self, state):
        self.ep_rew = 0
        self.delay = 0
        self.frame = 0
        self.num_of_episodes += 1
        self.present_result = 0
        self.present_st = self.getStateMatrices(state)
        self.prev_bonus = 0.
        self.prev_act = None
        self.prev_result = 0
        self.prev_st = None
        self.terminal = None
        self.World_Q = []
        self.won = True

    def train(self):
        if (self.class_count > self.parameter['train_start']):
            bitch = random.sample(self.memory_rep, self.parameter['batch_size'])
            b_s,b_r,b_a,b_n,b_t  = [],[],[],[],[]
             # = []
             # = []
             # = []
             # = []
            for i in bitch:
                b_s.append(i[0])
                b_r.append(i[1])
                b_a.append(i[2])
                b_n.append(i[3])
                b_t.append(i[4])
            b_a = self.obtain_1hot(np.array(b_a))
            b_n = np.array(b_n)
            b_r = np.array(b_r)
            b_s = np.array(b_s)
            b_t = np.array(b_t)
            self.count, self.val_d = self.nn.train(b_s, b_a, b_t, b_n, b_r)

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

        layer_name = 'conv3' ; size = 3 ; channels = 32 ; filters = 64 ; stride = 1
        self.w3 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c3 = tf.nn.conv2d(self.o2, self.w3, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o3 = tf.nn.relu(tf.add(self.c3,self.b3),name=self.network_name + '_'+layer_name+'_activations')

        layer_name = 'conv4' ; size = 3 ; channels = 64 ; filters = 128 ; stride = 1
        self.w4 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c4 = tf.nn.conv2d(self.o3, self.w4, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs')
        self.o4 = tf.nn.relu(tf.add(self.c4,self.b4),name=self.network_name + '_'+layer_name+'_activations')

        o2_shape = self.o4.get_shape().as_list()

        # Layer 3 (Fully connected)
        layer_name = 'fc3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
        self.o2_flat = tf.reshape(self.o4, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
        self.w5 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w5),self.b5,name=self.network_name + '_'+layer_name+'_ips')
        self.o5 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations')

        # Layer 4
        layer_name = 'fc4' ; hiddens = 4 ; dim = 256
        self.w6 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b6 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o5,self.w6),self.b6,name=self.network_name + '_'+layer_name+'_outputs')

        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))


        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # self.optim = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)
        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())


    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost
