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
            self.params = json.load(f)
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DQN(self.params)
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []
        self.cost_disp = 0
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0
        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.
        self.replay_mem = deque()
        self.last_scores = deque()


    def getMove(self, state):
        if np.random.rand() > self.params['eps']:
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            move = self.get_direction(np.random.randint(0, 4))
        self.last_action = self.get_value(move)
        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0
        elif direction == Directions.SOUTH:
            return 1
        elif direction == Directions.EAST:
            return 2
        else:
            return 3

    def get_direction(self, value):
        if value == 0:
            return Directions.NORTH
        elif value == 1:
            return Directions.SOUTH
        elif value == 2:
            return Directions.EAST
        else:
            return Directions.WEST

    def observation_step(self, state):
        if self.last_action is not None:
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score
            if reward > 20:
                self.last_reward = 50.
            elif reward > 0:
                self.last_reward = 10.
            elif reward < -10:
                self.last_reward = -500.
                self.won = False
            elif reward < 0:
                self.last_reward = -1.
            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()
            self.train()
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state):
        self.terminal = False
        self.observation_step(state)
        return state

    def final(self, state):
        self.ep_rew += self.last_reward
        self.terminal = True
        self.observation_step(state)
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f \n" %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.flush()

    def train(self):
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []
            batch_r = []
            batch_a = []
            batch_n = []
            batch_t = []
            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)
            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        width, height = self.params['width'], self.params['height']
        input_matrix = np.zeros((6, width, height))
        pacman_matrix = np.zeros((width, height))
        ghost_matrix = np.zeros((width, height))
        scared_matrix = np.zeros((width, height))
        capsule_matrix = np.zeros((width, height))
        input_matrix[0] = np.array(state.data.layout.walls.data, int)
        for i in state.data.agentStates:
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
        input_matrix[4] = np.array(state.data.food.data, int)
        for i in state.data.capsules:
            capsule_matrix[i[0]][i[1]] = 1
        input_matrix[5] = capsule_matrix
        return np.swapaxes(input_matrix, 0, 2)

    def registerInitialState(self, state):
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0
        self.last_state = None
        self.current_state = self.getStateMatrices(state)
        self.last_action = None
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move

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
