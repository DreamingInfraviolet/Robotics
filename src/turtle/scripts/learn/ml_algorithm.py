import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import random

class MLAlgorithm(object):
    pass

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states):
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Bottleneck in predict()
    def get_batch(self, model, batch_size=10):
        # Batch size can not exceed memory size
        batch_size = min(len(self.memory), batch_size)
        # Number of actions
        num_actions = model.output_shape[-1]
        # Number of inputs
        env_dim = self.memory[0][0][0].shape[1]

        # len(batch_size)*len(input) matrix
        inputs = np.zeros((batch_size, env_dim))
        # len(batch_size)*len(outputs) matrix
        targets = np.zeros((inputs.shape[0], num_actions))

        # Loop through batch size and populate inputs/targets
        for iBatchElement, iRandomMemory in enumerate(np.random.randint(0, len(self.memory), size=inputs.shape[0])):
            # Pick random memory
            state_t, action_t, reward_t, state_tp1 = self.memory[iRandomMemory][0]

            # Set input equal to start state of memory
            inputs[iBatchElement:iBatchElement+1] = state_t

            # Set target equal to predicted output weights
            # This way, if we are not changing a Q value for an input, it will not affect the NN for that input
            targets[iBatchElement] = model.predict(state_t)[0]

            # Find Q_sa, the maximum expected Q value from the resulting state
            Q_of_next_state = model.predict(state_tp1)[0]

            # action q value = reward + discount * max Q of resulting state
            targets[iBatchElement, action_t] = reward_t + self.discount * np.max(Q_of_next_state)
        return inputs, targets


# class ToyQlearningAlgorithm(MLAlgorithm):

#     class LearningData(object):
#         def __init__(self):
#             self.model = None
#             self.experienceReplay = None
#             self.actionCount = 0

#     class EpochData(object):
#         def __init__(self):
#             self.winCount = 0
#             self.loss = 0.0
#             self.initialInputs = None
#             self.actionTaken = -1

#     def __init__(self, explorationEpsilon, maxMemory, hiddenSize, batchSize):
#         self.epsilon    = explorationEpsilon
#         self.maxMemory  = maxMemory
#         self.hiddenSize = hiddenSize
#         self.batchSize  = batchSize

#         self.epochData = None
#         self.learningData = None

#     def startLearning(self, inputCount, actionCount):
#         self.learningData = self.LearningData()

#         self.learningData.actionCount = actionCount

#         self.learningData.model = Sequential()
#         self.learningData.model.add(Dense(self.hiddenSize, input_shape=(inputCount,), activation='relu'))
#         self.learningData.model.add(Dense(self.hiddenSize, activation='relu'))
#         self.learningData.model.add(Dense(self.learningData.actionCount))
#         self.learningData.model.compile(sgd(lr=.2), "mse")

#         self.learningData.experienceReplay = ExperienceReplay(max_memory=self.maxMemory)

#         # If you want to continue training from a previous model, just uncomment the line bellow
#         # model.load_weights("model.h5")

#     def startEpoch(self):
#         self.epochData = self.EpochData()

#     def decideOnAction(self, currentInputs):
#         currentInputs = np.array(currentInputs).reshape((1, -1))

#         # get next action
#         if np.random.rand() <= self.epsilon:
#             # Perform random action, explore!
#             action = np.random.randint(0, self.learningData.actionCount, size=1)
#         else:
#             # Predict best action to take
#             q = self.learningData.model.predict(currentInputs)
#             action = np.argmax(q[0])

#         self.epochData.initialInputs = currentInputs
#         self.epochData.actionTaken = action

#         return action

#     def reactToDecision(self, reward, newInputs, finalDecision):
#         self.learningData.experienceReplay.remember([self.epochData.initialInputs, self.epochData.actionTaken, reward, np.array(newInputs).reshape((1, -1))], finalDecision)
#         inputs, targets = self.learningData.experienceReplay.get_batch(self.learningData.model, batch_size=self.batchSize)
#         self.learningData.model.train_on_batch(inputs, targets)


#     def endEpoch(self):
#         pass


# class RealQlearningAlgorithm(MLAlgorithm):

#     class LearningData(object):
#         def __init__(self):
#             self.model = None
#             self.experienceReplay = None
#             self.actionCount = 0
#             self.epochIndex = 0

#     class EpochData(object):
#         def __init__(self):
#             self.winCount = 0
#             self.initialInputs = None
#             self.actionTaken = -1

#     def __init__(self, explorationEpsilon, epsilonDecay, maxMemory, hiddenSize, batchSize, temporalDiscount):
#         self.epsilon    = explorationEpsilon
#         self.epsilonDecay = epsilonDecay
#         self.maxMemory  = maxMemory
#         self.hiddenSize = hiddenSize
#         self.batchSize  = batchSize
#         self.temporalDiscount = temporalDiscount

#         self.epochData = None
#         self.learningData = None

#     def startLearning(self, inputCount, actionCount):
#         self.learningData = self.LearningData()

#         self.learningData.actionCount = actionCount
#         print(inputCount)
#         self.learningData.model = Sequential()
#         self.learningData.model.add(Dense(self.hiddenSize, input_shape=(inputCount,), activation='relu'))
#         self.learningData.model.add(Dense(self.hiddenSize, activation='relu'))
#         self.learningData.model.add(Dense(self.learningData.actionCount))
#         self.learningData.model.compile(keras.optimizers.Adamax(), "mse")

#         self.learningData.experienceReplay = ExperienceReplay(max_memory=self.maxMemory, discount=self.temporalDiscount)

#         # If you want to continue training from a previous model, just uncomment the line bellow
#         # model.load_weights("model.h5")

#     def startEpoch(self):
#         self.epochData = self.EpochData()
#         print("Training with epsilon = " + str(self.getCurrentExplorationEpsilon()))

#     def decideOnAction(self, currentInputs):
#         currentInputs = np.array(currentInputs).reshape((1, -1))

#         # get next action
        
#         if np.random.rand() <= self.getCurrentExplorationEpsilon():
#             # Perform random action, explore!
#             action = np.random.randint(0, self.learningData.actionCount, size=1)
#         else:
#             # Predict best action to take
#             q = self.QForAllActions(currentInputs)
#             action = np.argmax(q[0])
            
#             print(q)
#             print("---")

#         self.epochData.initialInputs = currentInputs
#         self.epochData.actionTaken = action

#         return action

#     def QForAllActions(self, inputs):
#         return self.learningData.model.predict(inputs)

#     def getCurrentExplorationEpsilon(self):
#         return self.epsilon ** (self.learningData.epochIndex * self.epsilonDecay)

#     def reactToDecision(self, reward, newInputs, finalDecision):
#         self.learningData.experienceReplay.remember([self.epochData.initialInputs, self.epochData.actionTaken, reward, np.array(newInputs).reshape((1, -1))])
#         inputs, targets = self.learningData.experienceReplay.get_batch(self.learningData.model, batch_size=self.batchSize)
#         # loss = self.learningData.model.train_on_batch(inputs, targets)
#         return 0 #loss

#     def endEpoch(self):
#         self.learningData.epochIndex = self.learningData.epochIndex + 1





########################### Policy gradient code

# ActorNetwork.py


import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from collections import deque
import random


import argparse
import json
import timeit



HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the actor model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4))(h1)  
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4))(h1) 
        Stuff = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4))(h1) 
        V = merge([Steering,Acceleration,Brake, Stuff],mode='concat')          
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S


# CriticNetwork.py

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the critic model")
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 


# OU.py

class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


# ReplayBuffer.py


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0




class PolicyGradient(MLAlgorithm):
    class LearningData(object):
        def __init__(self):
            self.state_dim = 0
            self.action_dim = 0
            self.actor = None
            self.critic = None
            self.buff = None
            self.firstRun = True
            self.epsilon = 1


    class EpochData(object):
        def __init__(self):
            self.s_t = None
            self.totalReward = 0
            self.a_t = None

    def __init__(self, sess):
        self.OU = OU()       #Ornstein-Uhlenbeck Process
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.TAU = 0.001     #Target Network HyperParameters
        self.LRA = 0.0001    #Learning rate for Actor
        self.LRC = 0.001     #Lerning rate for Critic
        self.EXPLORE = 100000.

        self.learningData = None
        self.epochData = None
        
        #1 means Train, 0 means simply Run
        self.train_indicator = 1

        self.sess = sess

    def startLearning(self, inputCount, actionCount):
        self.learningData = self.LearningData()
        self.epochData    = self.EpochData()

        self.learningData.state_dim = inputCount
        self.learningData.action_dim = actionCount
        self.learningData.actor = ActorNetwork(self.sess, self.learningData.state_dim, self.learningData.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.learningData.critic = CriticNetwork(self.sess, self.learningData.state_dim, self.learningData.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.learningData.buff = ReplayBuffer(self.BUFFER_SIZE)    #Create replay buffer
    
    def startEpoch(self):        
        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))q
        pass
        

    def decideOnAction(self, currentInputs):
        if self.learningData.firstRun:
            self.epochData.s_t = np.hstack(currentInputs)
            self.learningData.firstRun = False


        loss = 0
        self.learningData.epsilon -= 1.0 / self.EXPLORE
        a_t = np.zeros([1,self.learningData.action_dim])
        noise_t = np.zeros([1,self.learningData.action_dim])

        a_t_original = self.learningData.actor.model.predict(self.epochData.s_t.reshape(1, self.epochData.s_t.shape[0]))
        noise_t[0][0] = self.train_indicator * max(self.learningData.epsilon, 0) * self.OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
        noise_t[0][1] = self.train_indicator * max(self.learningData.epsilon, 0) * self.OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
        noise_t[0][2] = self.train_indicator * max(self.learningData.epsilon, 0) * self.OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        self.epochData.a_t = a_t

        return a_t[0]

    def reactToDecision(self, reward, newInputs, finalDecision):
        s_t1 = np.hstack(newInputs)
    
        self.learningData.buff.add(self.epochData.s_t, self.epochData.a_t[0], reward, s_t1, finalDecision)      #Add replay buffer
        
        #Do the batch update
        batch = self.learningData.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.learningData.critic.target_model.predict([new_states, self.learningData.actor.target_model.predict(new_states)])  
        
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.GAMMA*target_q_values[k]
    
        if (self.train_indicator):
            loss = self.learningData.critic.model.train_on_batch([states,actions], y_t) 
            a_for_grad = self.learningData.actor.model.predict(states)
            grads = self.learningData.critic.gradients(states, a_for_grad)
            self.learningData.actor.train(states, grads)
            self.learningData.actor.target_train()
            self.learningData.critic.target_train()

        self.epochData.totalReward += reward
        self.learningData.s_t = s_t1
    
        # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
    
        # step += 1
        return loss


    def endEpoch(self):
        pass

