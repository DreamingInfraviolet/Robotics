import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import adadelta
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


class RealQlearningAlgorithm(MLAlgorithm):

    class LearningData(object):
        def __init__(self):
            self.model = None
            self.experienceReplay = None
            self.actionCount = 0
            self.epochIndex = 0

    class EpochData(object):
        def __init__(self):
            self.winCount = 0
            self.loss = 0.0
            self.initialInputs = None
            self.actionTaken = -1

    def __init__(self, explorationEpsilon, epsilonDecay, maxMemory, hiddenSize, batchSize, temporalDiscount):
        self.epsilon    = explorationEpsilon
        self.epsilonDecay = epsilonDecay
        self.maxMemory  = maxMemory
        self.hiddenSize = hiddenSize
        self.batchSize  = batchSize
        self.temporalDiscount = temporalDiscount

        self.epochData = None
        self.learningData = None

    def startLearning(self, inputCount, actionCount):
        self.learningData = self.LearningData()

        self.learningData.actionCount = actionCount
        print(inputCount)
        self.learningData.model = Sequential()
        self.learningData.model.add(Dense(self.hiddenSize, input_shape=(inputCount,), activation='relu'))
        # self.learningData.model.add(Dense(self.hiddenSize, activation='tanh'))
        self.learningData.model.add(Dense(self.learningData.actionCount))
        self.learningData.model.compile(adadelta(), "mse")

        self.learningData.experienceReplay = ExperienceReplay(max_memory=self.maxMemory, discount=self.temporalDiscount)

        # If you want to continue training from a previous model, just uncomment the line bellow
        # model.load_weights("model.h5")

    def startEpoch(self):
        self.epochData = self.EpochData()
        print("Training with epsilon = " + str(self.getCurrentExplorationEpsilon()))

    def decideOnAction(self, currentInputs):
        currentInputs = np.array(currentInputs).reshape((1, -1))

        # get next action
        
        if np.random.rand() <= self.getCurrentExplorationEpsilon():
            # Perform random action, explore!
            action = np.random.randint(0, self.learningData.actionCount, size=1)
        else:
            # Predict best action to take
            q = self.QForAllActions(currentInputs)
            action = np.argmax(q[0])
            
            print(q)
            print("---")

        self.epochData.initialInputs = currentInputs
        self.epochData.actionTaken = action

        return action

    def QForAllActions(self, inputs):
        return self.learningData.model.predict(inputs)

    def getCurrentExplorationEpsilon(self):
        return self.epsilon ** (self.learningData.epochIndex * self.epsilonDecay)

    def reactToDecision(self, reward, newInputs, finalDecision):
        self.learningData.experienceReplay.remember([self.epochData.initialInputs, self.epochData.actionTaken, reward, np.array(newInputs).reshape((1, -1))])
        inputs, targets = self.learningData.experienceReplay.get_batch(self.learningData.model, batch_size=self.batchSize)
        loss = self.learningData.model.train_on_batch(inputs, targets)
        print("Trained with loss " + str(loss))

    def endEpoch(self):
        self.learningData.epochIndex = self.learningData.epochIndex + 1